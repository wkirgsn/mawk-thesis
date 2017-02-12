#!/usr/bin/env python

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mawk.trainer import Trainer
from mawk.datascript import DataPack
import argparse
import os
from os.path import join, splitext
from os import listdir
from math import ceil, sqrt
import pandas as pd


def add_shared_args(parser):
    parser.add_argument('--workingplace', '-wp',
                        choices=DataPack.workingplaces.keys(),
                        default='nt', help='Place where you work at this code')
    parser.add_argument('--nohdf', action='store_true',
                        help='Use NZP instead of HDF5 (h5py)')
    parser.add_argument('--root', '-r', default='',
                        help='Root directory path of training/test data')
    parser.add_argument('--toy', action='store_true',
                        help='Use just a small subset of the data')


def plot_trend(_trainer):
    train_trend = np.trim_zeros(_trainer.recorder[0, :], trim='b')
    val_trend = np.trim_zeros(_trainer.recorder[1, :], trim='b')
    xticks = np.arange(1, len(train_trend)+1, 1)
    plt.figure()
    plt.plot(xticks, train_trend, label='training')
    plt.plot(xticks, val_trend, label='validation')
    plt.axvline(np.argmin(val_trend) + 1, color='r', ls='--')
    plt.xticks(xticks[4::5], [str(s) for s in xticks[4::5]])  # ticks at 5,10,15,20 ...
    plt.ylabel('{} normalized'.format(_trainer.lossfun))
    plt.xlabel('epochs')
    plt.title(_trainer.id + ' training trend')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(_trainer.out_path,
                             'traintrend_' + _trainer.id + '.svg'),
                format='svg')


def plot_net_prediction(groundtruth, prediction, title='',
                        savepath='', trainer_id=''):
    time = np.arange(prediction.shape[1], dtype=np.float32)
    time /= (2*60)
    #prediction = prediction[2, :].reshape([1, -1])
    #groundtruth = groundtruth[2, :].reshape([1, -1])
    if prediction.shape[0] == 1:
        m = 1
        n = 1
    elif prediction.shape[0] < 3:
        m = 1
        n = 2
    else:
        m = 2
        n = 2
    w = n*5
    h = w * (sqrt(5)-1.0)/2.0

    plt.figure(title, figsize=(w, h))
    for out in range(prediction.shape[0]):
        plt.subplot(m, n, out+1)
        plt.plot(time, prediction[out, :], label='Prediction')
        plt.plot(time, groundtruth[out, :], label='Groundtruth')
        plt.xlabel('time in minutes')
        plt.ylabel('Temp. in °C')
        #plt.tick_params(axis='both', which='major', pad=10)
        #plt.title(trainer_.data.targets_tup._fields[out])
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2,
                   mode='expand', borderaxespad=0.0)
        #print('mean max. dev: {} °C'.format(
         #   np.abs(prediction[out, :] - groundtruth[out, :].max())))
    plt.tight_layout()
    if savepath != '':
        plt.savefig(os.path.join(savepath,
                                 trainer_id + '_prediction_' + title + '.svg'),
                    format='svg')


def plot_train_data(_data, train_data, time_factor=(2*60), num="",
                    mark_outliers=False, profile_markers=None, benchdata=True,
                    anomalies=None):
    """Plot data with optional markers.

    Args:
        _data (DataPack): The instance to fetch param names from.
        train_data (np.ndarray): The data to display. Shape:[n_params, length]
        time_factor:
        num:
        mark_outliers:
        profile_markers:
        benchdata:
        anomalies:

    """
    time = np.arange(train_data.shape[1], dtype=np.float32)
    time /= time_factor
    if profile_markers is not None:
            profile_markers[0, :] /= time_factor
    plt.figure(num)
    for out in range(train_data.shape[0]):
        s = train_data[out, :]
        if not train_data.shape[0] == 1:
            plt.subplot(ceil(float(train_data.shape[0]) / 2), 2, out+1)
        plt.plot(time, s)
        plt.xlim(xmax=max(time))
        if benchdata:
            plt.xlabel('time in minutes')
        if not _data.preprocess[0] == 'pca':
            plt.title(_data.Inputs_tup._fields[out]
                      if out < len(_data.Inputs_tup._fields)
                      else
                      _data.Targets_tup._fields[out -len(_data.Inputs_tup._fields)])
        else:
            plt.title('Input') if out < _data.n_input_params else \
                plt.title('Output')
        if mark_outliers:
            window = 40
            indices_to_mark = []
            corrected_points = []
            for point in range(s.size):
                min_p = max(point - window/2, 0)
                max_p = min(point + window/2, s.size-1)
                windowed_frame = s[min_p:max_p] - s[min_p:max_p].mean()
                if np.absolute(s[point] - s[min_p:max_p].mean()) > \
                                windowed_frame.std() * 3:
                    indices_to_mark.append(point)
                    corrected_points.append(np.median(s[min_p:max_p]))
            plt.plot(time[indices_to_mark], s[indices_to_mark], 'go')
            plt.plot(time[indices_to_mark], corrected_points, 'rd')
        if profile_markers is not None:
            [plt.axvline(x, color='g', ls='--') for x in profile_markers[0, :]]
            nr_offset = (profile_markers[0, :][1] - profile_markers[0, :][0])/50
            [plt.text(profile_markers[0, :][p] + nr_offset,
                      max(s),
                      int(profile_markers[1, :][p]), verticalalignment='top')
             for p in range(len(profile_markers[0, :]))]
        if anomalies is not None:
            for p in anomalies[out]:
                plt.plot(time[p[0]:p[1]], s[p[0]:p[1]],
                         c='red',
                         lw=2.0)
    plt.show()


def setup_vis_trainer(args):
    """Setup visualizing Trainer class"""
    args.initmodel = True
    args.gpu = -1
    if args.out[-1] == '/':
        args.out = args.out[:-1]
    return Trainer(args, visualizing_only=True)


def forward_through_all_sets(_trainer, plot=True, quiet=False,
                             train_only=False):
    if not train_only:
        sets = {'val': [_trainer.val_pool, _trainer.data.n_val],
                'test': [_trainer.test_pool, _trainer.data.n_test]}
    else:
        trainpool, _, _ = \
            _trainer.data.load_all_profiles_as_batches(toy=_trainer.toy)
        sets = {'train': [trainpool, _trainer.data.n_train]}

    test_trend = np.trim_zeros(_trainer.recorder[1, :], trim='b')
    ret = {'best_net_after_epochs': 1 + np.argmin(test_trend),
           'best_net_after_time': _trainer.recorder[2, np.argmin(
                                   test_trend)],
           'total_epochs': test_trend.shape[0],
           'total_time': _trainer.recorder[2, test_trend.shape[0] - 1],
           }


    # Print duration stats
    print('Best net trained after '
          '{} epochs ({:.2f} h). Total: {} epochs ({:.2f} h)'.format(
            ret['best_net_after_epochs'],
            ret['best_net_after_time'],
            ret['total_epochs'],
            ret['total_time']))

    # forward through all sets
    for key, value in sets.items():
        pool = value[0]
        ndata = value[1]
        mean_loss, real_loss, groundtruth_un, prediction_un = \
            _trainer.evaluation_loop(pool, ndata)
        if quiet:
            ret[key + '_mean_loss_norm'] = mean_loss if not np.isnan(
                mean_loss) else 1e10
            if isinstance(real_loss, float):
                ret[key + '_mean_loss_real'] = 1e10
                for a in range(_trainer.data.n_target_params):
                    ret[key + '_real_' + _trainer.data.targets_tup._fields[a]] = \
                        1e10
            else:
                ret[key + '_mean_loss_real'] = np.vstack(real_loss).mean() if \
                    not any(np.isnan(f) for f in real_loss) else 1e10
                for a in range(_trainer.data.n_target_params):
                    ret[key + '_real_' + _trainer.data.targets_tup._fields[a]] = \
                        real_loss[a]
        else:
            # print performance
            print('[' + key + '] mean loss {:12} = {:.6f} '
                  .format('(normalized)', mean_loss))
            print('[' + key + '] mean loss {:12} = {:.4f} K^2 <<<<'
                  .format('(real)', np.vstack(real_loss).mean()))
            [print('[' + key + '] mean loss for {:14}: {:.4f} K^2'.format(
                _trainer.data.targets_tup._fields[a],
                real_loss[a])) for a in
                range(_trainer.data.n_target_params)]
            if plot:
                plot_net_prediction(groundtruth_un, prediction_un,
                                    title=key,
                                    savepath=_trainer.out_path,
                                    trainer_id=_trainer.id)
                # save to hdf
                for x, lb in [(groundtruth_un, 'groundtruth'),
                              (prediction_un, 'prediction')]:
                    df = pd.DataFrame(x.T,
                                      columns=
                                      [_trainer.data.targets_tup._fields[out]
                                      for out in range(x.shape[0])])
                    df.to_hdf(join(_trainer.out_path, key+'.prediction'), lb)

    if plot:
        plot_trend(_trainer)
        plt.show()
    if quiet:
        return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MAWK visualization utility')
    add_shared_args(parser)
    parser.add_argument('--out', '-o', default='results',
                        help='Path to saved model and state')
    parser.add_argument('--benchdata', '-b',
                        choices=DataPack.Inputs_tup._fields +
                                DataPack.Targets_tup._fields + ('all', ),
                        default=None,
                        help='Plot benchmark data only')
    parser.add_argument('--train_only', action='store_true',
                        help='This flag will evaluate performance on the '
                             'train subset only rather than on validation and'
                             ' test subset')
    parser.add_argument('--write', '-w', action='store_true',
                        help='Write performance into an .ini-file')
    parser.add_argument('--load', '-l', action='store_true',
                        help='Load prediction and groundtruth form HDF5')
    args = parser.parse_args()

    matplotlib.rcParams['svg.fonttype'] = 'none'  # for latex pics

    if args.benchdata:
        # show benchmark dataq
        data = DataPack(args.workingplace, np) \
                if args.root == '' else DataPack(args.root, np)
        in_dict, _ = \
            data.load_all_profiles_simple(toy=args.toy)
        x_train = in_dict['train']

        if args.benchdata == 'all':
            # plot all profiles
            p_all = [np.vstack(p) for p in x_train]
        else:
            # plot specific value
            p_all = [p._asdict()[args.benchdata] for p in x_train]
        bd = True if args.benchdata else False
        plot_train_data(data, np.hstack(p_all), benchdata=bd)
    else:
        # Setup visualizing trainer
        trainer = setup_vis_trainer(args)
        if args.write:
            import configparser
            ret_dict = forward_through_all_sets(trainer, plot=False,
                                                quiet=True,
                                                train_only=args.train_only)
            config = configparser.ConfigParser()
            config['EvaluationResults'] = ret_dict
            inipath = os.path.join(trainer.out_path,
                                   'evaluation_result_' + trainer.id + '.ini')
            with open(inipath, 'w+') as inifile:
                config.write(inifile)

        else:
            if any(x.endswith('.prediction') for x in listdir(
                    trainer.out_path)) and args.load:
                # prediction available
                for set_lb in ['val', 'test']:
                    ext = '.prediction'
                    groundtruth = pd.read_hdf(join(trainer.out_path, set_lb +
                                                   ext),
                                              'groundtruth').as_matrix().T
                    prediction = pd.read_hdf(join(trainer.out_path, set_lb +
                                                  ext),
                                             'prediction').as_matrix().T

                    plot_net_prediction(groundtruth, prediction,
                                        set_lb, trainer_id=trainer.id)
                plt.show()
            else:
                forward_through_all_sets(trainer, train_only=args.train_only)
