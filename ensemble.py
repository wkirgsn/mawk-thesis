#!/usr/bin/env python

import argparse
from os.path import join, split, isfile, isdir
from os import listdir, makedirs
import pandas as pd
from visuals import setup_vis_trainer, plot_net_prediction
import numpy as np
import matplotlib.pyplot as plt
from mawk.trainer import Trainer
from chainer.functions.loss import mean_squared_error as MSE


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run ensemble. Takes the '
                                                 'best n models of a specific '
                                                 'pso run')
    parser.add_argument('--hdf', '-f',
                        help='Path to evaluations hdf file')
    parser.add_argument('--run', '-r',
                        help='Path to iterations of that run being ensembled')
    parser.add_argument('--n_models', '-n', type=int,
                        help='Amount of models to ensemble')
    args = parser.parse_args()

    # remove trailing slash
    if args.run.endswith('/'):
        args.run = args.run[:-1]
    if args.hdf.endswith('/'):
        args.hdf = args.hdf[:-1]

    # get best models and create list of their paths
    run_id = split(args.run)[1]
    plot_other_predictions = False if 'Run1' in run_id or 'Run2' in run_id \
        else True
    df = pd.read_hdf(args.hdf, key=run_id)
    smallest_df = df['test_score'].nsmallest(n=args.n_models)
    smallest_df = df.loc[df['test_score'].isin(smallest_df),
                         ['iter', 'p_id', 'test_score']]
    smallest_df.sort_values(by='test_score', ascending=True, inplace=True)
    print(smallest_df)
    model_paths = []
    for n in range(args.n_models):
        path = join(args.run, 'iter_' + smallest_df.iloc[n, 0],
                    smallest_df.iloc[n, 1])
        model_paths.append(path)
    for i, m in enumerate(model_paths):
        for f in listdir(m):
            if isdir(join(m, f)):
                model_paths[i] = join(m, f)
                break

    trainer_ids = [split(m)[1] for m in model_paths]
    with pd.HDFStore(join(split(args.hdf)[0], 'predictions')) as store:
        backup_available = True
        for trainer_id in trainer_ids:
            if run_id + '_' + trainer_id not in store:
                backup_available = False
        if run_id + '_groundtruth' not in store:
            backup_available = False

        if backup_available:
            print('load backup of predictions:')
            predictions = []
            for trainer_id in trainer_ids:
                predictions.append(store[run_id + '_' +
                                         trainer_id].as_matrix().T)
            groundtruth = store[run_id + '_groundtruth'].as_matrix().T
        else:
            print('No predictions backed up. Start evaluating models:')

            trainer_list = []
            args.workingplace = 'tp'
            args.toy = False
            args.root = ''
            args.nohdf = False
            for path in model_paths:
                args.out = path
                trainer_list.append(setup_vis_trainer(args))
            predictions = []
            groundtruth = None
            for trainer in trainer_list:
                _, _, groundtruth_, prediction = trainer.evaluation_loop(
                    trainer.test_pool, trainer.data.n_test)
                predictions.append(prediction)
                groundtruth = groundtruth_

            # save predictions to hdf
            trainers = zip(trainer_list, predictions)
            cols = [trainer.data.targets_tup._fields[o]
                    for o in range(predictions[0].shape[0])]
            for trainer, pred in trainers:
                store[run_id + '_' + trainer.id] = \
                    pd.DataFrame(pred.T, columns=cols)
            store[run_id + '_groundtruth'] = pd.DataFrame(groundtruth.T,
                                                          columns=cols)
    best_loss = 300
    for i in range(1, 1+len(predictions)):
        new_prediction = np.dstack(predictions[:i]).mean(axis=2)
        loss = Trainer.compute_loss(new_prediction.shape[0], new_prediction,
                                    groundtruth, MSE.mean_squared_error)
        loss = np.asarray(loss).mean()
        max_dev = np.max(np.abs(new_prediction - groundtruth))
        if loss < best_loss:
            best_loss = loss
            best_loss_at = i
            best_prediction = new_prediction
            best_max_dev = max_dev
        print('Ensemble: {} model(s) test error: {} K², max_dev: {} '
              '°C'.format(i, loss, max_dev))

    if plot_other_predictions:
        plt.figure('ensemble')
        time = np.arange(best_prediction.shape[1], dtype=np.float32)

        time /= (2*60)
        plt.plot(time, best_prediction.ravel(), label='ensemble', lw=2.0)
        plt.plot(time, groundtruth.ravel(), label='groundtruth', lw=2.0)
        for i, pred in enumerate(predictions[:best_loss_at]):
            plt.plot(time, pred.ravel(), label=str(i))
        plt.xlabel('time in minutes')
        plt.ylabel('Temp. in °C')
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=5,
                   mode='expand', borderaxespad=0.0)
    else:
        plot_net_prediction(groundtruth, best_prediction)
        print('individual temperatures of best ensemble:')
        max_dev = np.max(np.abs(new_prediction - groundtruth), axis=1)
        loss = Trainer.compute_loss(best_prediction.shape[0], best_prediction,
                                    groundtruth, MSE.mean_squared_error)
        labels = ['pm', 'yoke', 'teeth', 'winding']
        for i in range(1, 1+len(loss)):
            print('{}: {:.4} K², maxdev: {} °C'.format(labels[-i], loss[-i],
                                                       max_dev[-i]))
    plt.show()
