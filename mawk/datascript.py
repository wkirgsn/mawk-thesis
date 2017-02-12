import scipy.io as sio
import numpy as np
from collections import namedtuple
import os.path
import sys
import chainer
import itertools
from sklearn.decomposition import PCA
from random import shuffle, randint
import time


class DataPack(object):
    workingplaces = {'nt': os.path.join('/home', 'wilhelmk', 'wilhelmk', 'MA',
                                        'Lastprofile', 'v7_preproccd'),
                     'lap': os.path.join('C:\\', 'Users', 'Wilhelm',
                                         'Documents', 'Uni_PB', 'MA',
                                         'Messdaten', 'Lastprofile',
                                         'v7_preproccd'),
                     'heiden': os.path.join('C:\\', 'Users', 'wkirc_000',
                                            'Documents', 'dev', 'data',
                                            'v7_preproccd'),
                     'tp': os.path.join('/home', 'wilhelmk', 'Messdaten',
                                        'PMSM_Lastprofile', 'v7_preproccd'),
                     'pc2': os.path.join('/upb', 'departments', 'pc2',
                                         'scratch', 'wilhelmk', 'data',
                                         'v7_preproccd')
                     }

    Input_param_names = ['ambient', 'vorlauf', 'ud_mdl', 'uq_mdl',
                         'drehzahl', 'T_ist', 'id_ist', 'iq_ist',
                         'i_nondq', 'u_nondq', 'psi']
    Target_param_names = ['pm', 'statorjoch', 'statorzahn', 'statorwicklung']

    Inputs_tup = namedtuple('Inputs', Input_param_names )
    Targets_tup = namedtuple('Targets', Target_param_names)

    def __init__(self, root, xp, outmask=15):
        self.xp = xp
        self.data_path = ""
        self.train_input = []
        self.train_output = []
        self.n_train = 0
        self.n_val = 0
        self.n_test = 0
        self.n_input_params = 0
        self.n_target_params = 0
        self.n_batches = 0
        self.batchsize = 0
        self.seq_len = 0
        self.train_subseqs = {}
        self.preprocess = []
        self.pca_x = PCA(whiten=True)  # only used when preprocess = pca
        self.pca_y = PCA(whiten=True)  # only used when preprocess = pca
        self.lookback = 0
        self.out_mask = [int(p) for p in format(outmask, '04b')]
        # Exclude profile 1, 22 and 25 because they are not realistic
        self.lowdyns = [2, 3, 5, 7, 8, 9] + list(range(12, 20)) + \
                       [21, 23, 24, 26]  # lowdyns = S1-heat-ups
        self.middyns = [4, 6, 10]
        self.highdyns = [11, 20] + list(range(27, 37))
        self.highdyns_350V_udc = [28, 33, 34, 35]
        self.trainset = []
        self.valset = []
        self.testset = []

        if all(self.out_mask):
            self.targets_tup = self.Targets_tup
        else:
            self.targets_tup = \
                namedtuple(
                        'Targets',
                        [self.Target_param_names[i] for i, k in
                         enumerate(self.out_mask) if k == 1])

        self.g_mean_std = np.zeros([2, len(self.Input_param_names) +
                                    len(self.targets_tup._fields)])
        # Exclude higher udc profiles
        [self.highdyns.pop(self.highdyns.index(x))
         for x in self.highdyns_350V_udc]

        # Determine path to data
        if root in self.workingplaces:
            print('Load data from previously configured workingplace -> '+root)
            self.data_path = self.workingplaces[root]
        else:
            print('Load data from ' + root)
            self.data_path = root

    def load_profiles(self, index_tuple, preprocessed_files=True):
        input_data = []
        target_data = []

        for k in range(len(index_tuple)):
            p_no = str(index_tuple[k]) if index_tuple[k] > 9 else \
                '0' + str(index_tuple[k])
            loadprofile = sio.loadmat(
                os.path.join(self.data_path, 'part0' + p_no))
            input = self.Inputs_tup(ambient=loadprofile['thetaS15'].astype(
                                        dtype=np.float32),
                                    vorlauf=loadprofile['thetaS16'].astype(
                                        dtype=np.float32),
                                    ud_mdl=loadprofile['ud_mdl'].astype(
                                        dtype=np.float32),
                                    uq_mdl=loadprofile['uq_mdl'].astype(
                                        dtype=np.float32),
                                    drehzahl=loadprofile['nme'].astype(
                                        dtype=np.float32),
                                    T_ist=loadprofile['Tx'].astype(
                                        dtype=np.float32),
                                    id_ist=loadprofile['idx'].astype(
                                        dtype=np.float32),
                                    iq_ist=loadprofile['iqx'].astype(
                                        dtype=np.float32),
                                    u_nondq=np.sqrt(
                                        np.square(loadprofile['ud_mdl'].astype(
                                            dtype=np.float32)) +
                                        np.square(loadprofile['uq_mdl'].astype(
                                            dtype=np.float32))),
                                    i_nondq=np.sqrt(
                                        np.square(loadprofile['idx'].astype(
                                            dtype=np.float32)) +
                                        np.square(loadprofile['iqx'].astype(
                                            dtype=np.float32))),
                                    psi=loadprofile['idx'].astype(
                                        dtype=np.float32) - np.divide(
                                        np.square(loadprofile['iqx'].astype(
                                            dtype=np.float32)),
                                        loadprofile['idx'].astype(
                                            dtype=np.float32)))
            # Future Work
            # udc=loadprofile['udc'].astype(dtype=np.float32),
            """ REDUNDANT
            lim_T_soll=loadprofile['Tw_lim'].astype(
                dtype=np.float32),
            id_soll=loadprofile['idw'].astype(
                dtype=np.float32),
            iq_soll=loadprofile['iqw'].astype(
                dtype=np.float32),

            ud_soll=loadprofile['udw'].astype(
                dtype=np.float32),
            uq_soll=loadprofile['uqw'].astype(
                dtype=np.float32),
                """

            if not preprocessed_files:
                pm_mean = (loadprofile['thetaR5'].astype(dtype=np.float32) +
                           loadprofile['thetaR6'].astype(dtype=np.float32) +
                           loadprofile['thetaR7'].astype(dtype=np.float32) +
                           loadprofile['thetaR8'].astype(dtype=np.float32))\
                          / 4.0
            else:
                pm_mean = loadprofile['thetaRmean'].astype(dtype=np.float32)

            t = (pm_mean,
                 loadprofile['thetaS03'].astype(dtype=np.float32),
                 loadprofile['thetaS07'].astype(dtype=np.float32),
                 loadprofile['thetaS09'].astype(dtype=np.float32))

            target = self.targets_tup._make([t[i] for i, x in
                                            enumerate(self.out_mask) if x == 1])
            input_data.append(input)
            target_data.append(target)

        return input_data, target_data

    def load_all_profiles_simple(self, valset=(31,), testset=(20, ), toy=False):

        assert self.preprocess[0] in ('normalize', 'pca'), \
            'Choose "normalize" or "pca" as preprocessing step in ' \
            'your config file'

        # Exclude S1-Heat-Up Profiles
        self.trainset = self.middyns + self.highdyns
        self.valset = valset
        self.testset = testset

        [self.trainset.pop(self.trainset.index(x)) for x in valset + testset]

        if toy:
            self.trainset = [self.trainset[0], ]
            self.valset = [self.valset[0], ]
            self.testset = [self.testset[0], ]

        x_train, y_train = self.load_profiles(self.trainset)
        x_val, y_val = self.load_profiles(self.valset)
        x_test, y_test = self.load_profiles(self.testset)

        assert len(x_train) == len(y_train)
        assert len(x_val) == len(y_val)
        assert len(x_test) == len(y_test)

        self.n_input_params = len(x_train[0])
        self.n_target_params = len(y_train[0])

        # Determine global mean and std per field (within training subset)
        self._investigate_global_characteristics(x_train, y_train)

        # Normalize data fieldwise
        x_train = self._normalize_data(x_train, 'in')
        x_val = self._normalize_data(x_val, 'in')
        x_test = self._normalize_data(x_test, 'in')
        y_train = self._normalize_data(y_train, 'out')
        y_val = self._normalize_data(y_val, 'out')
        y_test = self._normalize_data(y_test, 'out')

        # Augment data with stoch. moments
        if self.lookback > 0:
            # This turns these lists' elements from namedtuples into ndarrays
            x_train = self.append_additional_data(x_train)
            x_val = self.append_additional_data(x_val)
            x_test = self.append_additional_data(x_test)

            self.n_input_params = x_train[0].shape[1]

        # PCA
        if self.preprocess[0] == 'pca':
            x_list = self._convert_namedtup2matrix(x_train)
            y_list = self._convert_namedtup2matrix(y_train)
            x_temp = np.vstack(x_list)
            y_temp = np.vstack(y_list)

            self.pca_x.fit(x_temp)
            self.pca_y.fit(y_temp)

            ratio = 0
            for n, comp in enumerate(self.pca_x.explained_variance_ratio_):
                ratio += comp
                if ratio > self.preprocess[1]:
                    self.pca_x = PCA(n_components=n+1)
                    self.pca_x.fit(x_temp)
                    break
            """ratio = 0   # Dont reduce output dimensions
            for n, comp in enumerate(self.pca_y.explained_variance_ratio_):
                ratio += comp
                if ratio > self.preprocess[1]:
                    self.pca_y = PCA(n_components=n+1)
                    self.pca_y.fit(y_temp)
                    break"""

            x_train = [self.pca_x.transform(p) for p in x_list]
            y_train = [self.pca_y.transform(p) for p in y_list]
            x_val = [self.pca_x.transform(p) for p in
                     self._convert_namedtup2matrix(x_val)]
            y_val = [self.pca_y.transform(p) for p in
                     self._convert_namedtup2matrix(y_val)]
            x_test = [self.pca_x.transform(p) for p in
                      self._convert_namedtup2matrix(x_test)]
            y_test = [self.pca_y.transform(p) for p in
                      self._convert_namedtup2matrix(y_test)]

            self.n_input_params = x_train[0].shape[1]
            self.n_target_params = y_train[0].shape[1]

        inputs = {'train': x_train, 'val': x_val, 'test': x_test}
        outputs = {'train': y_train, 'val': y_val, 'test': y_test}

        return inputs, outputs

    def load_all_profiles_as_batches(self, toy=False):

        data_setup_start = time.time()

        # Load normalized data
        in_dict, out_dict = self.load_all_profiles_simple(toy=toy)
        self.train_input = self._convert_namedtup2matrix(in_dict['train'])
        self.train_output = self._convert_namedtup2matrix(out_dict['train'])

        # Convert to chainer variables
        # Trainset
        x_train_doublelist, y_train_doublelist = \
            self.shuffle_train_data(self.train_input, self.train_output)
        # Val- and testset
        x_val_doublelist = self._divide_into_batches(in_dict['val'], 'on')
        y_val_doublelist = self._divide_into_batches(out_dict['val'], 'on')
        x_test_doublelist = self._divide_into_batches(in_dict['test'], 'on')
        y_test_doublelist = self._divide_into_batches(out_dict['test'], 'on')

        # Count amounts
        self.n_train = sum([len(b)*b[0].data.shape[0] for b in
                            x_train_doublelist])
        self.n_val = sum([len(b)*b[0].data.shape[0] for b in
                          x_val_doublelist])
        self.n_test = sum([len(b)*b[0].data.shape[0] for b in
                           x_test_doublelist])

        # Print data stats
        print("\n{} input parameters; {} output parameters".format(
            self.n_input_params, self.n_target_params))
        print('train data = {:.2%} ({}) - '.format(
            self.n_train/(self.n_train + self.n_test + self.n_val),
            self.n_train) +
            'val data = {:.2%} ({}) - '.format(
            self.n_val/(self.n_train + self.n_test + self.n_val), self.n_val) +
            'test data = {:.2%} ({})'.format(
            self.n_test/(self.n_train + self.n_test + self.n_val), self.n_test))
        print('data setup duration: {:.1f} seconds\n'.format(time.time() -
                                                             data_setup_start))
        # return pools as dicts
        train_pool = {'x': x_train_doublelist, 'y': y_train_doublelist}
        val_pool = {'x': x_val_doublelist, 'y': y_val_doublelist}
        test_pool = {'x': x_test_doublelist, 'y': y_test_doublelist}
        return train_pool, val_pool, test_pool

    def _divide_into_batches(self, pool,  volatileflag='auto', batchsize=1):
        datapool = self._convert_namedtup2matrix(pool)
        # profile in datapool: (seqlen, n_params)
        batches = []
        if batchsize == 1:
            for p in datapool:
                profile_specific_batches = []
                for i in range(p.shape[0]):
                    profile_specific_batches.append(
                        chainer.Variable(p[i, :].reshape([1, -1]),
                                         volatile=volatileflag))
                batches.append(profile_specific_batches)
        else:
            # arbitrary batchsize
            for p in datapool:
                profile_specific_batches = []
                seqlen = p.shape[0]
                jump = seqlen // batchsize
                for i in range(jump):
                    minibatch = np.vstack([p[jump * j + i, :] for j in
                                           range(batchsize)])
                    profile_specific_batches.append(
                        chainer.Variable(minibatch, volatile=volatileflag)
                    )
                batches.append(profile_specific_batches)

        return batches

    def _investigate_global_characteristics(self, i_all_data, o_all_data,
                                            quiet=True):
        """Calculates mean and std in given data and saves them in
        self.g_mean_std

        Args:
            i_all_data: input data (trainset only)
            o_all_data: output data (trainset only)
            quiet: print calculated values or not

        """
        n_in = self.n_input_params
        n_out = self.n_target_params
        for field_id in range(n_in + n_out):
            id = field_id if field_id < n_in else field_id - n_in
            fieldname = self.Input_param_names[id] \
                if field_id < n_in \
                else self.targets_tup._fields[id]
            p_collector = []
            for p in range(len(i_all_data)):
                stack = np.vstack(i_all_data[p]) if field_id < n_in else \
                    np.vstack(o_all_data[p])
                p_collector.append(stack[id, :])
            s_stack = np.hstack(p_collector)
            self.g_mean_std[0, field_id] = np.mean(s_stack)
            self.g_mean_std[1, field_id] = np.std(s_stack)
            if not quiet:
                print(fieldname)
                print(self.g_mean_std[0, field_id])
                print(self.g_mean_std[1, field_id])

    def append_additional_data(self, pool):
        """Appends stochastic moments to the given pools.

        Args:
            pool: List of pools to be extended

        Returns: Augmented pools

        """
        pool = self._convert_namedtup2matrix(pool)
        # now: pool_element.shape == (profile_len, n_params)
        newpool = []
        for p in pool:
            # stoch. moments
            means = np.zeros(shape=p.shape, dtype=np.float32)
            variances = np.zeros(shape=p.shape, dtype=np.float32)

            for col in range(p.shape[0]):
                means[col, :] = np.mean(p[max(0, col - self.lookback):col+1, :],
                                        axis=0)
                variances[col, :] = \
                    np.var(p[max(0, col - self.lookback):col+1, :], axis=0)

            newpool.append(np.hstack([p, means, variances]))
        return newpool

    @staticmethod
    def _divide_into_subseqs(pool, subseqlen, start=None):
        """Divides pool into chunks of lists of length subseqlen"""
        # profile.shape in datapool: (seqlen, n_params)
        sub_sequences = []
        shorter_seqs = []
        for i, p in enumerate(pool):
            assert p.shape[0] > subseqlen,\
                'Profile is too short. Shape[0] must be at least equal to ' \
                'subseqlen'
            if start is None:
                start_idx = randint(0, p.shape[0] % subseqlen)
            else:
                start_idx = start[i]
            subs = [p[j:j+subseqlen, :] for j in range(start_idx, p.shape[0],
                                                       subseqlen)]
            if not subs[-1].shape[0] == subseqlen:
                shorter_seqs.append(subs.pop())
            sub_sequences.append(subs)
        return sub_sequences, shorter_seqs

    @staticmethod
    def _convert_namedtup2matrix(*args):
        """Convert namedtuple to numpy matrix"""
        p = args[0]
        # check if list or matrix
        if type(p) == list:
            if not type(p[0]) == np.ndarray:
                return [np.asarray(np.vstack(a).T) for a in p]
        else:
            if not type(p) == np.ndarray:
                return np.asarray(np.vstack(p).T)
        # type(p) == nampedtuple
        return p

    @staticmethod
    def shuffle_pairwise(x, y):
        """Shuffles two lists, x and y, the same way and order."""
        comb = list(zip(x, y))
        shuffle(comb)
        x[:], y[:] = zip(*comb)

    def shuffle_train_data(self, pool_in, pool_out):
        """Takes normalized data and returns chainer Variables accordingly"""
        # Determine start idx for each profile for both pools to hold on to
        start = [randint(0, p.shape[0] % self.seq_len) for p in pool_in]
        assert len(pool_in) == len(pool_out)

        # Divide normalized data into subsequences
        self.train_subseqs['x'], self.train_subseqs['x_short'] = \
            self._divide_into_subseqs(pool_in, self.seq_len, start)
        self.train_subseqs['y'], self.train_subseqs['y_short'] = \
            self._divide_into_subseqs(pool_out, self.seq_len, start)

        # Flatten Lists
        flat_x = list(itertools.chain.from_iterable(self.train_subseqs['x']))
        flat_y = list(itertools.chain.from_iterable(self.train_subseqs['y']))
        assert len(flat_x) == len(flat_y)
        assert flat_x[0].shape[0] == self.seq_len
        assert flat_y[0].shape[0] == self.seq_len

        # Shuffle pairwise x<->y
        self.shuffle_pairwise(flat_x, flat_y)

        # create profiles with size = batchsize, converted to chainer Vars
        minibatch_x = [flat_x[i:i+self.batchsize] for i in
                       range(0, len(flat_x), self.batchsize)]
        minibatch_y = [flat_y[i:i+self.batchsize] for i in
                       range(0, len(flat_y), self.batchsize)]
        assert len(minibatch_x) == len(minibatch_y)

        for m, n in zip(minibatch_x, minibatch_y):
            m[:] = [chainer.Variable(np.vstack([p[i, :] for p in m])) for
                    i in range(min(self.seq_len, m[0].shape[0]))]
            n[:] = [chainer.Variable(np.vstack([p[i, :] for p in n])) for
                    i in range(self.seq_len)]

        return minibatch_x, minibatch_y

    def _normalize_data(self, List_to_norm, tag=''):
        normed_list = []
        for p in range(len(List_to_norm)):
            normed_profile = []
            anz = self.n_input_params if tag == "in" else self.n_target_params
            for f in range(anz):
                idx = f if tag == "in" else f+self.n_input_params
                tup = List_to_norm[p]
                fieldarr = tup[f]
                normed_profile.append(
                       np.divide(
                               fieldarr - self.g_mean_std[0, idx].astype(
                                       np.float32),
                               self.g_mean_std[1, idx].astype(np.float32)))
            if tag == 'in':
                normed_list.append(self.Inputs_tup._make(normed_profile))
            elif tag == 'out':
                normed_list.append(self.targets_tup._make(normed_profile))
        return normed_list

    def unnormalize_profile(self, p):
        """Unnormalize profile data.

        Take normalized ndarray and unnormalize it according to saved mean
        and standard deviation in DataPack-class.
        Converts output parameters only.

        Args:
            p (np.ndarray): profile. shape: (n_outputs, profile_len)

        Returns: (n_outputs, profile_len) ndarray of unnormalized data

        """
        if self.preprocess[0] == 'pca':
            p = self.pca_y.inverse_transform(p.T).T

        unnormalized_p = \
            np.multiply(p,
                        self.g_mean_std[1, -self.n_target_params:]
                        .reshape((self.n_target_params, 1))) + \
            self.g_mean_std[0, -self.n_target_params:].reshape(
                    (self.n_target_params, 1))

        return unnormalized_p

    def _preprocess_data_and_save(self):
        """ This method was run one time to remove outliers"""
        input_keys = ['thetaS15', 'thetaS16', 'ud_mdl', 'uq_mdl', 'nme', 'Tx',
                      'udc', 'idx', 'iqx']
        output_keys = ['thetaRmean', 'thetaS03', 'thetaS07', 'thetaS09']
        keys_to_save = [input_keys, output_keys]
        all_indices = self.highdyns_350V_udc + self.lowdyns + self.middyns + \
                      self.highdyns

        i_all_data, o_all_data = self.load_profiles(all_indices,
                                                    preprocessed_files=True)
        window = 40
        # iterate through all profiles
        for p in all_indices:
            profile = {}
            i_stack, o_stack = np.vstack(i_all_data[all_indices.index(p)]),\
                               np.vstack(o_all_data[all_indices.index(p)])
            stacks = [i_stack, o_stack]
            for stack_no in range(len(stacks)):
                for field_id in range(stacks[stack_no].shape[0]):
                    s = stacks[stack_no][field_id, :]
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
                    s[indices_to_mark] = corrected_points
                    stacks[stack_no][field_id, :] = s
                for key_id_no in range(len(keys_to_save[stack_no])):
                    profile[keys_to_save[stack_no][key_id_no]] = \
                        stacks[stack_no][key_id_no, :]
            # save
            p_no = str(p) if p > 9 else '0' + str(p)
            path_to_save_to = os.path.join('/home', 'wilhelmk', 'wilhelmk','MA',
                                         'Lastprofile', 'v7_preproccd_3')
            if not os.path.isdir(path_to_save_to):
                os.makedirs(path_to_save_to)
            sio.savemat(os.path.join(path_to_save_to, 'part0' + p_no),
                        profile)


