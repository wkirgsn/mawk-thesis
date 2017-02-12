from chainer import cuda
import chainer
from chainer import serializers
import scipy.io as sio
import mawk.net_collection as net_collection
from mawk.datascript import DataPack
import numpy as np
import os
from random import shuffle
import configparser
import uuid
import time


def add_args(parser):
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--workingplace', '-wp',
                        choices=DataPack.workingplaces.keys(),
                        default='nt', help='Place where you work at this code')
    parser.add_argument('--root', '-r', default='',
                        help='Root directory path of training/test data')
    parser.add_argument('--out', '-o', default='results',
                        help='Path to save model at during training')
    parser.add_argument('--initmodel', default='',
                        help='Initialize the model from given file')
    parser.add_argument('--resume', default='',
                        help='Resume the optimization from snapshot')
    parser.add_argument('--plot', action='store_true',
                        help='Plot performance after training')
    parser.add_argument('--nohdf', action='store_true',
                        help='Use NZP instead of HDF5 (h5py)')
    parser.add_argument('--toy', action='store_true',
                        help='Use just a small subset of the data')
    parser.add_argument('--no_val', action='store_true',
                        help='Skip validating nets on val set during training')
    parser.add_argument('--config', '-c',
                        default=os.path.join('settings', 'general_config.ini'),
                        help='Path to config file')


class Trainer(object):

    def __init__(self, args, visualizing_only=False, datapack=None):
        config = configparser.ConfigParser()
        if visualizing_only:
            self.id = os.path.split(args.out)[1]
            args.config = os.path.join(args.out, self.id + '.ini')
            filename = config.read(args.config)
            if len(filename) == 0:
                raise EnvironmentError('Config file {} in {} not '
                                       'found'.format(args.config, args.out))
            print('Trainer ID: {}'.format(self.id))
            self.out_path = args.out
        else:
            filename = config.read(args.config)
            if len(filename) == 0:
                raise EnvironmentError('Given config file not found: '
                                       '{}'.format(args.config))
            while True:
                # generate unique id
                self.id = str(uuid.uuid4()).split(sep='-')[0]
                # Set result out path
                self.out_path = os.path.join(args.out, self.id)
                if os.path.isdir(self.out_path):
                    print('ID {} exists already! Generate a new one.'.format(
                        self.id))
                else:
                    print('New Trainer ID: {}'.format(self.id))
                    break
            # create result directory
            os.makedirs(self.out_path)
            # Copy config file to out path
            with open(os.path.join(self.out_path, self.id + '.ini'), 'w') as fn:
                config.write(fn)

        # Net architecture settings
        netconf = config['Net']
        self.arch = netconf['arch']
        self.n_hl = int(netconf['n_hl'])
        self.n_units = int(netconf['n_units'])
        mask = int(netconf['outmask'])
        w_init_distrib = netconf['weight_init_distribution']
        w_init_scaling = netconf['weight_init_scaling']

        # Training settings
        self.toy = args.toy
        trainconf = config['Training']
        self.overfit = int(trainconf['earliest_early_stop'])
        self.n_epochs = int(trainconf['epochs'])
        seq_len = int(trainconf['seq_len'])
        batchsize = int(trainconf['batchsize'])
        preprocess = [trainconf['preprocess'], ]
        if preprocess[0] == 'pca':
            preprocess.append(float(trainconf['pca_var_ratio_to_keep']))
        lookback = int(trainconf['lookback'])

        # Optimization settings
        optconf = config['Opt']
        self.lossfun = optconf['lossfunc']
        optimizer_name = optconf['optimizer']
        self.lr_init = np.float32(optconf['lr_init'])
        self.lr_decay = np.float32(optconf['lr_decay'])

        # Regularization settings
        regconf = config['Regularization']
        dropout = False

        active = [int(p) for p in
                  format(int(regconf['active']), '02b')]
        keys = sorted(['gaussnoise', 'weightdecay'])
        assert len(active) == len(keys), "{} != {}".format(len(active),
                                                           len(keys))
        [regconf.pop(keys[i]) for i, r in enumerate(active) if r == 0]

        regs = []
        for k in net_collection.reg_dict.keys():
            if not regconf[k] == 'no':
                regs += [k, regconf[k]]

        # Trend-Recorder
        self.recorder = np.zeros([3, self.n_epochs], np.float32)

        self.fnn_flashback = 0 if 'fnn' in self.arch else 0
        self.xp = np if args.gpu < 0 else cuda.cupy

        if datapack is None:
            # load data
            self.data =\
                DataPack(args.workingplace, xp=self.xp, outmask=mask)\
                if args.root == '' else \
                DataPack(args.root, xp=self.xp, outmask=mask)
        else:
            self.data = datapack
        self.data.preprocess = preprocess
        self.data.batchsize = batchsize
        self.data.seq_len = seq_len
        self.data.lookback = lookback
        self.train_pool, self.val_pool, self.test_pool = \
            self.data.load_all_profiles_as_batches(toy=args.toy)

        # setup model and optimizer
        model_and_opt_specs = {'architecture': self.arch,
                               'n_in': self.data.n_input_params,
                               'n_out': self.data.n_target_params,
                               'n_hl': self.n_hl,
                               'n_units': self.n_units,
                               'wdistribution': w_init_distrib,
                               'wscaleheuristic': w_init_scaling,
                               'loss_name': self.lossfun,
                               'optim_name': optimizer_name,
                               'lr_init': self.lr_init,
                               'train_mode': dropout,
                               'regularization': regs}
        self.model, self.optimizer = \
            net_collection.get_model_and_optimizer(**model_and_opt_specs)

        self.nohdf = args.nohdf
        # load model from save_path if desired
        if args.initmodel:
            self.load_from_file()

        if args.gpu >= 0:
            cuda.get_device(args.gpu).use()
            self.model.to_gpu()

        self.best_net = {'net': self.model,
                         'state': self.optimizer,
                         'loss': np.inf}
        self.sum_loss = 0
        self.prediction = []

    def training_loop(self, quiet=True):
        self.sum_loss = 0

        # Shuffle all subsequences from all profiles
        self.train_pool['x'], self.train_pool['y'] = \
            self.data.shuffle_train_data(self.data.train_input,
                                         self.data.train_output)
        b = 0
        b_last = 0
        start = time.time()
        # loop through profiles
        for x_p, t_p in zip(self.train_pool['x'], self.train_pool['y']):

            accum_loss = 0
            if self.arch in ['lstm', 'gru', 'lstm_peep']:
                    self.model.predictor.reset_state()

            # process batch
            for x, t in zip(x_p, t_p):
                loss_i = self.model(x, t)
                accum_loss += loss_i
                self.sum_loss += loss_i.data * len(t.data)
                b += 1

            self.model.zerograds()
            accum_loss.backward()
            self.optimizer.update()

            if not quiet:
                b_diff = b - b_last
                time_diff = time.time() - start
                print('proceed batch nr '
                      '{}/{}. rate = {} batches per minute'.format(
                        b, self.data.n_batches,
                        (60*b_diff)/time_diff))
                start = time.time()
                b_last = b
        """ # Gradient debugging:
            if len(self.optimizer._hooks) > 0:
                  print('mean: {}'.format(self.optimizer._hooks[
                            'GradientDisplaying'].avg_norm()))"""
        _mean_loss = self.sum_loss / self.data.n_train
        return _mean_loss

    def evaluation_loop(self, pool, n_data, quiet=True):
        self.sum_loss = 0
        self.prediction.clear()
        evaluator = self.model.copy()
        evaluator.predictor.train = False

        b = 0
        for x_p, t_p in zip(pool['x'], pool['y']):
            if self.arch in ['lstm', 'gru', 'lstm_peep']:
                evaluator.predictor.reset_state()
            for x, t in zip(x_p, t_p):
                if not quiet:
                    b += 1
                    print('proceed batch nr {}/{}'.format(b, len(pool['x'])))
                loss = evaluator(x, t)
                self.sum_loss += float(loss.data) * len(t.data)
                self.prediction.append(evaluator.y.data)

        _mean_loss = self.sum_loss / n_data

        if np.isnan(_mean_loss):
            print('Mean loss became NaN! Abort')
            return np.nan, np.nan, np.nan, np.nan

        # unnormalize
        _groundtruth_un, _prediction_un = \
            self.get_groundtruth_and_prediction(pool['y'])
        _real_loss = self.compute_loss(self.data.n_target_params,
                                       _prediction_un, _groundtruth_un,
                                       evaluator.lossfun)

        return _mean_loss, _real_loss, _groundtruth_un, _prediction_un

    @staticmethod
    def compute_loss(n_target_params, prediction, target, lossfun):
        """Compute loss between prediction and target.

        :param prediction: Predicted time series.
            shape: (n_targets, len(profile))
        :type prediction: np.ndarray
        :param target: (n_target_params, time_series_length) ndarray
        :return: List of unnormalized loss for every target parameter
        """
        return [float(lossfun(
                chainer.Variable(prediction[i].astype(np.float32),
                                 volatile='on'),
                chainer.Variable(target[i].astype(np.float32),
                                 volatile='on')
                ).data)
                for i in range(n_target_params)]

    def get_groundtruth_and_prediction(self, pool):
        """This function works only for 1-layered pools (no partly stacked
        pools)

        Args:
            pool (list): The data to compare the attribute 'prediction' with.

        Returns:
            Groundtruth and prediction unnormalized

        """
        groundtruth = np.vstack([b.data for p in pool for b in p]).T
        prediction = np.vstack(self.prediction).T

        groundtruth_un = self.data.unnormalize_profile(groundtruth)
        prediction_un = self.data.unnormalize_profile(prediction)

        return groundtruth_un, prediction_un

    def adjust_lr(self, decrease):
        print('Decrease lr to ') if decrease else print('Increase lr to ')
        factor = self.lr_decay if decrease else 1 + (1 - self.lr_decay)
        if isinstance(self.optimizer, chainer.optimizers.Adam):
            self.optimizer.alpha *= factor
            print(self.optimizer.alpha)
        elif isinstance(self.optimizer, chainer.optimizers.AdaDelta):
            self.optimizer.rho *= factor
            print(self.optimizer.rho)
        else:
            self.optimizer.lr *= factor
            print(self.optimizer.lr)

    def save_to_file(self, serialize=False):
        # save train trend
        trend = {'train': self.recorder[0, :],
                 'val': self.recorder[1, :],
                 'dur': self.recorder[2, :]
                 }
        sio.savemat(os.path.join(self.out_path, '{}_trend'.format(self.id)),
                    trend)

        # save model and optimizer
        if serialize:
            # Save the model and the optimizer
            save = serializers.save_npz if self.nohdf else serializers.save_hdf5
            save(os.path.join(self.out_path, '{}.model'.format(self.id)),
                 self.model)
            save(os.path.join(self.out_path, '{}.state'.format(self.id)),
                 self.optimizer)

    def load_from_file(self, load_state=False):
        print('Load training trend ' + os.path.join(self.out_path,
                                                    self.id+'_trend.mat'))
        trend = sio.loadmat(os.path.join(self.out_path,
                                         self.id+'_trend'))
        self.recorder = np.vstack([trend['train'],
                                   trend['val'],
                                   trend['dur']
                                   ])
        print('Load model ' + os.path.join(self.out_path, self.id))
        load = serializers.load_npz if self.nohdf else serializers.load_hdf5
        load(os.path.join(self.out_path, self.id+'.model'), self.model)
        if load_state:
            print('Load optimizer state from', self.out_path)
            load(os.path.join(self.out_path, self.id+'.state'),
                 self.optimizer)
