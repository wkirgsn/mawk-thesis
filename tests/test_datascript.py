import unittest

from mawk.datascript import DataPack
import numpy as np


class TestDataPack(unittest.TestCase):
    def setUp(self):
        self.data = DataPack('tp', np)
        self.data.preprocess = ['normalize', ]
        self.data.batchsize = 16
        self.data.seq_len = 50
        self.indict, self.outdict = self.data.load_all_profiles_simple()

    def test_divide_into_subseqs(self):
        n_params = 4
        base_matrix = np.arange(100).reshape([-1, n_params])
        pool = [np.vstack(
            [np.add(base_matrix, np.random.normal(scale=0.1,
                                                  size=base_matrix.shape))
             for i in range(161)]) for j in range(9)]
        subseq_len = 2030
        subs1, shorter_subs1 = DataPack._divide_into_subseqs(pool, subseq_len)
        subs2, shorter_subs2 = DataPack._divide_into_subseqs(pool, subseq_len)

        for s1, s2 in zip(subs1, subs2):
            for s1_i, s2_i in zip(s1, s2):
                self.assertEqual(s1_i.shape, s2_i.shape)
                self.assertEqual(s1_i.shape, (subseq_len, n_params))
                self.assertFalse(np.array_equal(s1_i, s2_i))
        for s1, s2 in zip(shorter_subs1, shorter_subs2):
            self.assertLess(len(s1), subseq_len)
            self.assertLess(len(s2), subseq_len)
            for s1_i, s2_i in zip(s1, s2):
                self.assertEqual(s1_i.shape, s2_i.shape)
                self.assertFalse(np.array_equal(s1_i, s2_i))

    def test_corresponding_xy_after_shuffle(self):
        n_params = 8
        factor = 2.5
        x_train = [np.arange(start=i, stop=2240+i, dtype=np.float32).reshape([
            -1, n_params]) for i in range(9)]
        y_train = [a * factor for a in x_train]
        x_new, y_new = self.data.shuffle_train_data(x_train, y_train)
        for x, y in zip(x_new, y_new):
            for x_i, y_i in zip(x, y):
                for m, n in zip(np.nditer(x_i.data), np.nditer(y_i.data)):
                    self.assertEqual(m*factor, n)

    def test_namedtup2matrix_converting(self):
        a = self.indict['train']
        b = self.data._convert_namedtup2matrix(a)
        c = [self.data._convert_namedtup2matrix(p) for p in a]
        for bb, cc in zip(b, c):
            self.assertTrue(np.allclose(bb, cc, 1e-6))

    def test_normalize_unnormalize(self):
        _, train_raw_y = self.data.load_profiles(self.data.trainset)
        _, val_raw_y = self.data.load_profiles(self.data.valset)
        _, test_raw_y = self.data.load_profiles(self.data.testset)

        train_raw_y = [self.data._convert_namedtup2matrix(p) for p
                       in train_raw_y]
        val_raw_y = [self.data._convert_namedtup2matrix(p) for p
                       in val_raw_y]
        test_raw_y = [self.data._convert_namedtup2matrix(p) for p
                       in test_raw_y]

        train_output_dbl_proccd = [self.data.unnormalize_profile(a.T).T for a
                                   in [
                            self.data._convert_namedtup2matrix(p) for p in
                            self.outdict['train']]]

        val_output_dbl_proccd = [self.data.unnormalize_profile(a.T).T for a in [
                            self.data._convert_namedtup2matrix(p) for p in
                            self.outdict['val']]]

        test_output_dbl_proccd = [self.data.unnormalize_profile(a.T).T for a
                                  in [
                            self.data._convert_namedtup2matrix(p) for p in
                            self.outdict['test']]]
        [self.assertTrue(np.allclose(m, n, 1e-6)) for m, n in zip(
            train_raw_y, train_output_dbl_proccd)]
        [self.assertTrue(np.allclose(m, n, 1e-6)) for m, n in zip(
            val_raw_y, val_output_dbl_proccd)]
        [self.assertTrue(np.allclose(m, n, 1e-6)) for m, n in zip(
            test_raw_y, test_output_dbl_proccd)]


