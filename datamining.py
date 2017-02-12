#!/usr/bin/env python


from mawk.datascript import DataPack
import argparse
import numpy as np
import pandas as pd
from os import listdir
from os.path import join, split, splitext
import scipy.io as sio
import random
from visuals import plot_train_data
from sklearn.metrics import mean_squared_error as mse
from sklearn import gaussian_process
from sklearn.linear_model import ARDRegression
import matplotlib.pyplot as plt
from math import ceil

param_map = {'thetaS15': 'ambient',
             'thetaS16': 'coolant',
             'ud_mdl': 'u_d',
             'uq_mdl': 'u_q',
             'nme': 'motor_speed',
             'Tx': 'torque',
             'idx': 'i_d',
             'iqx': 'i_q',
             'thetaRmean': 'pm',
             'thetaS03': 'stator_yoke',
             'thetaS07': 'stator_tooth',
             'thetaS09': 'stator_winding'
             }
symbols_map = {'ambient': '$\\vartheta_a$\n in °C',
               'coolant': '$\\vartheta_c$\n in °C',
               'u_d': '$u_d$\n in V',
               'u_q': '$u_q$\n in V',
               'motor_speed': '$n_{mech}$\n in 1/min',
               'torque': '$T_x$\n in Nm',
               'i_d': '$i_d$\n in A',
               'i_q': '$i_q$\n in A',
               'pm': '$\\vartheta_{PM}$\n in °C',
               'stator_yoke': '$\\vartheta_{SY}$\n in °C',
               'stator_tooth': '$\\vartheta_{ST}$\n in °C',
               'stator_winding': '$\\vartheta_{SW}$\n in °C'
               }
loadsets = [4, 6, 10, 11, 20, 27, 29, 30, 31, 32, 36]


def setup_data():
    data = DataPack(args.workingplace, xp=np) if args.root == '' else \
           DataPack(args.root, xp=np)
    data.preprocess = ['normalize', ]
    #data.preprocess = ['pca', 0.93]
    indict, outdict = data.load_all_profiles_simple(valset=(), testset=())
    pool =\
        [np.hstack([data._convert_namedtup2matrix(indict['train'][p]),
                    data._convert_namedtup2matrix(outdict['train'][p])]).T
            for p in range(len(indict['train']))]
    return data, pool


def setup_data_matfile_2_csv_hdf5(matfile_root):
    filenames = sorted(listdir(matfile_root),
                       key=lambda k: int(splitext(k)[0][-3:]))

    matfiles = [sio.loadmat(join(matfile_root, f)) for f in
                filenames if int(splitext(f)[0].endswith(
                 tuple(['{:03d}'.format(a) for a in loadsets])))]
    load_ids, columns = zip(*param_map.items())
    dframes = [pd.DataFrame(np.hstack([m[k].T for k in load_ids]),
                            columns=columns) for m in matfiles]
    hdf_dir = join(split(matfile_root)[0], 'hdf')
    hdf_file = join(hdf_dir, 'all_load_profiles')
    csv_dir = join(split(matfile_root)[0], 'csv')
    for df, p_no in zip(dframes, loadsets):
        df.to_hdf(hdf_file, 'p' + str(p_no))
        df.to_csv(join(csv_dir, str(p_no)), index=False)


def load_benchmark_data_from_hdf(hdf_file):
    return [pd.read_hdf(hdf_file, key='p'+str(k)) for k in loadsets]


def load_benchmark_data_from_csv(csv_root):
    return [pd.read_csv(join(csv_root, l)) for l in listdir(csv_root)]


def analyze_sensitivity(frames):

    ard = ARDRegression(compute_score=True)

    plt.figure()
    plt.plot(ard.coef_, 'b-', label='ARD estimate')
    #plt.xticks(range(len(inputs)), inputs, rotation=30)
    plt.xlabel('Features')
    plt.ylabel('Values of the weights?')
    plt.legend(loc=1)
    plt.show()


def determine_relevance():
    df = pd.read_hdf(join(head, 'hdf', 'evaluations'), key=tail)
    df.apply(lambda k: (k - np.mean(k)) / (np.max(k) - np.min(k)))
    y = df[['test_score', 'val_score']]
    x = df.drop(['test_score', 'val_score'], axis=1)
    samples = random.sample(range(x.shape[0]), 100)
    gp = gaussian_process.GaussianProcessRegressor()
    gp = gp.fit(x.iloc[samples], y.iloc[samples])

    print('debug')


def plot_normalized_profiles_from_hdf(hdf_file_path):
    df_list = load_benchmark_data_from_hdf(hdf_file_path)
    load_lengths = [d.shape[0] for d in df_list]
    profilemarks = \
        np.asanyarray([[sum(load_lengths[:i]) for i, p in
                        enumerate(load_lengths)],
                       loadsets], dtype=float)
    df = pd.concat(df_list)
    time = np.arange(df.shape[0], dtype=float) / (2*60)
    profilemarks[0, :] /= (2*60)
    plt.figure()
    for out in range(df.shape[1]):
        s = df[df.columns[out]]
        ax = plt.subplot(df.shape[1], 1, out+1)
        ax.yaxis.set_label_coords(-0.1, 0.5)
        plt.plot(time, s)
        plt.xlim(xmax=max(time))
        if out == df.shape[1]-1:
            plt.xlabel('time in minutes')
        else:
            plt.tick_params(labelbottom='off')
        plt.ylabel(symbols_map[df.columns[out]], rotation='horizontal')
        ymin, ymax = plt.ylim()
        plt.yticks([ymin+0.1*(ymax-ymin), ymin+0.9*(ymax-ymin)])
        nr_offset = (profilemarks[0, :][1] - profilemarks[0, :][0])/40
        for idx, mark in enumerate(profilemarks[0, :]):
            plt.axvline(mark, color='g', ls='--')
            if out == 0:
                plt.text(mark + nr_offset, ymax + 0.3*(ymax-ymin),
                         int(profilemarks[1, idx]), verticalalignment='top')
        if out == 0:
            plt.title('Profile no.', y=1.3)
    plt.show()


def plot_normalized_profiles(_data, _pool):
    """Plot plain pool data with profile separators

    Args:
        _data (DataPack): The instance to fetch the param names from
        _pool (List): The data to plot. List (len=n_profiles) of np arrays of
            shape:[n_params, len(profile)]

    """
    profilemarks = get_profilemarks(_data, _pool)
    plot_train_data(_data, np.hstack(_pool), profile_markers=profilemarks)


def get_profilemarks(_data, _pool):
    """Extract indices for vertical profile separators.

    Args:
        _data (DataPack): The instance to get profile_IDs from.
        _pool (List): List of profiles. Each profile contains np.ndarrays of
            data to plot (shape:[n_params, len(profiles)]).

    Returns:
        2d-Array of shifted indices for profile separators (
        first row) and corresponding profile IDs (second row). shape:
        (2, n_profiles).

    """
    profilemarks = np.asanyarray([[p.shape[1] for p in _pool], _data.trainset],
                                 dtype=np.float32)
    profilemarks[0, :] = np.asanyarray([sum(profilemarks[0, :p]) for p in
                                        range(profilemarks.shape[1])])
    return profilemarks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Datamining PMSM profiles')
    parser.add_argument('--root', '-r', default='',
                        help='Root directory path to training/test data')
    args = parser.parse_args()

    head, tail = split(args.root)
    #benchmark_data = '/home/wilhelmk/Messdaten/PMSM_Lastprofile/v7_preproccd'
    #setup_data_matfile_2_csv_hdf5(benchmark_data)
    plot_normalized_profiles_from_hdf(args.root)
    #data, pool = setup_data()
    #plot_normalized_profiles(data, pool)



