import matplotlib
import matplotlib.pyplot as plt
from os import listdir, makedirs
from os.path import join, split, isfile, isdir
from math import sqrt
import argparse
import configparser
import numpy as np
import pandas as pd
from cycler import cycler
from pso import position2param, get_hyperparam_bounds, hyper_param_intervals,\
    hyper_param_intervals_2


def collect_results(resultspath):
        configreader = configparser.ConfigParser()
        best_test_mean = {'mse': np.inf,
                          'p_id': None,
                          'iter': 0}
        best_test_pm = best_test_mean.copy()
        best_test_statorjoch = best_test_mean.copy()
        best_test_statorzahn = best_test_mean.copy()
        best_test_statorwicklung = best_test_mean.copy()

        best_val_mean = best_test_mean.copy()
        best_val_pm = best_test_mean.copy()
        best_val_statorjoch = best_test_mean.copy()
        best_val_statorzahn = best_test_mean.copy()
        best_val_statorwicklung = best_test_mean.copy()

        columns = []
        for category in hyper_param_intervals:
            for hyper_param_name in hyper_param_intervals[category]:
                columns.append(hyper_param_name)
        columns.append('test_score')
        columns.append('val_score')
        columns.append('p_id')
        columns.append('iter')

        iter_paths = sorted(listdir(resultspath),
                            key=lambda h: int(h.split('_')[-1]))

        # init hyperparameterset/score map
        evaluations = {}
        for h in columns:
            evaluations[h] = []

        for iteration_path in iter_paths:
            iteration_no = iteration_path.split('_')[-1]
            print('analyze iteration {}'.format(iteration_no))
            for p_id in listdir(join(resultspath, iteration_path)):
                rc = configreader.read(join(resultspath, iteration_path, p_id,
                                            'particle_info_' + p_id + '.ini'))
                if len(rc) == 0:
                    print('Could not read particle info in iter '
                          '{} for particle {}'.format(iteration_no, p_id))
                else:
                    results = configreader['EvaluationResults']
                    d_test = {'mean': [best_test_mean,
                              float(results['test_mean_loss_real'])]}

                    if configreader.has_option('EvaluationResults',
                                               'test_real_pm'):
                        d_test['pm'] = [best_test_pm,
                                float(results['test_real_pm'])]
                    if configreader.has_option('EvaluationResults',
                                               'test_real_statorjoch'):
                        d_test['statorjoch'] = [best_test_statorjoch,
                                float(results['test_real_statorjoch'])]
                    if configreader.has_option('EvaluationResults',
                                               'test_real_statorzahn'):
                        d_test['statorzahn'] = [best_test_statorzahn,
                                float(results['test_real_statorzahn'])]
                    if configreader.has_option('EvaluationResults',
                                               'test_real_statorwicklung'):
                        d_test['statorwicklung'] = [best_test_statorwicklung,
                                float(results['test_real_statorwicklung'])]

                    d_val = {'mean': [best_val_mean,
                                      float(results['val_mean_loss_real'])]}
                    if configreader.has_option('EvaluationResults',
                                               'val_real_pm'):
                        d_val['pm'] = [best_val_pm,
                                       float(results['val_real_pm'])]
                    if configreader.has_option('EvaluationResults',
                                               'val_real_statorjoch'):
                        d_val['statorjoch'] = [best_val_statorjoch,
                                               float(results[
                                               'val_real_statorjoch'])]
                    if configreader.has_option('EvaluationResults',
                                               'val_real_statorzahn'):
                        d_val['statorzahn'] = [best_val_statorzahn,
                                               float(results[
                                               'val_real_statorzahn'])]
                    if configreader.has_option('EvaluationResults',
                                               'val_real_statorwicklung'):
                        d_val['statorwicklung'] = [best_val_statorwicklung,
                                                   float(results[
                                                   'val_real_statorwicklung'])]

                    for k, v in d_test.items():
                        if v[1] < v[0]['mse']:
                            d_test[k][0]['mse'] = v[1]
                            d_test[k][0]['p_id'] = p_id
                            d_test[k][0]['iter'] = iteration_no
                    for k, v in d_val.items():
                        if v[1] < v[0]['mse']:
                            d_val[k][0]['mse'] = v[1]
                            d_val[k][0]['p_id'] = p_id
                            d_val[k][0]['iter'] = iteration_no

                    # update hyperparamset/score results
                    position = configreader['Position']
                    for h in columns:
                        if h not in ['test_score', 'val_score', 'iter', 'p_id']:
                            evaluations[h].append(float(position[h]))
                    evaluations['test_score'].append(float(d_test['mean'][1]))
                    evaluations['val_score'].append(float(d_val['mean'][1]))
                    evaluations['iter'].append(iteration_no)
                    evaluations['p_id'].append(p_id)

        print('\nBest Test Performance:')
        d_test = {'mean': best_test_mean}
        if int(best_test_pm['iter']) > 0:
            d_test['pm'] = best_test_pm
        if int(best_test_statorjoch['iter']) > 0:
            d_test['statorjoch'] = best_test_statorjoch
        if int(best_test_statorzahn['iter']) > 0:
            d_test['statorzahn'] = best_test_statorzahn
        if int(best_test_statorwicklung['iter']) > 0:
            d_test['statorwicklung'] = best_test_statorwicklung
        for k, v in d_test.items():
            for a in v:
                print('{}| {}: {}'.format(k, a, v[a]))

        print('\nBest Validation Performance:')
        d_val = {'mean': best_val_mean}
        if int(best_val_pm['iter']) > 0:
            d_val['pm'] = best_val_pm
        if int(best_val_statorjoch['iter']) > 0:
            d_val['statorjoch'] = best_val_statorjoch
        if int(best_val_statorzahn['iter']) > 0:
            d_val['statorzahn'] = best_val_statorzahn
        if int(best_val_statorwicklung['iter']) > 0:
            d_val['statorwicklung'] = best_val_statorwicklung
        for k, v in d_val.items():
            for a in v:
                print('{}| {}: {}'.format(k, a, v[a]))

        # save hyperparamset/score results
        for h in columns:
            evaluations[h] = \
                np.asarray(evaluations[h]).reshape((len(evaluations[h]), 1))
        df = pd.DataFrame(np.hstack([evaluations[h] for h in columns]),
                          columns=columns)
        for c in df.columns:
            if c not in ['p_id', 'iter']:
                df[c] = df[c].apply(lambda x: np.float(x))
        upper_dir, run_id = split(resultspath)
        target_path = join(upper_dir, 'hdf')
        df.to_hdf(join(target_path, 'evaluations'), run_id)


def plot_convergence(resultspath, hyper=1):
    configreader = configparser.ConfigParser()
    iter_paths = sorted(listdir(resultspath),
                        key=lambda k: int(k.split('_')[-1]))

    hyper_intervals = hyper_param_intervals_2 if hyper == 2 else\
        hyper_param_intervals

    # init counters
    eval_per_iter = []
    gbest_per_iter = np.zeros([len(iter_paths), ])
    vel_per_iter = []
    param_dists = {}
    var_per_iter = {}
    for k, v in hyper_intervals.items():
        for kk, vv in v.items():
            param_dists[kk] = [vv[0], ]
            var_per_iter[kk] = []

    # iterate through pso iterations
    for iter_idx, i in enumerate(iter_paths):
        iteration_no = i.split('_')[-1]
        print('analyze iteration {}'.format(iteration_no))
        assert int(iteration_no) == iter_idx + 1

        # init sub counters
        iter_evals = []
        iter_vels = []
        iter_vars = {}
        for k in var_per_iter:
            iter_vars[k] = []
        gbest_per_iter[iter_idx] = np.inf if iter_idx == 0 else \
            gbest_per_iter[iter_idx - 1]

        # iterate through particles
        for p_id in listdir(join(resultspath, i)):
            configreader.clear()
            # collect fitness
            rc = configreader.read(join(resultspath, i,
                                        p_id, 'fitness.ini'))
            if len(rc) == 0:
                print('Could not read fitness.ini in iter '
                      '{} for particle {}'.format(iteration_no, p_id))
            else:
                fitness_sec = configreader['Fitness']
                curr = float(fitness_sec['current'])
                if curr < 1000:
                    iter_evals.append(curr)
                gbest = float(fitness_sec['global_best'])
                if gbest < gbest_per_iter[iter_idx]:
                    gbest_per_iter[iter_idx] = gbest

            # collect particle info
            rc = configreader.read(join(resultspath, i,
                                        p_id, 'particle_info_' + p_id +
                                        '.ini'))
            if len(rc) == 0:
                print('Could not read current_settings.ini in iter '
                      '{} for particle {}'.format(iteration_no, p_id))
            else:
                # collect velocities
                vel_sec = dict(configreader['Velocity'])
                elementwise_vels = []
                for k, v in vel_sec.items():
                    for u_key, l_dict in hyper_intervals.items():
                        if k in l_dict.keys():
                            # check distribution
                            dist = l_dict[k][0]
                            if dist == 'xor':
                                interval = len(l_dict[k][1:])
                            else:
                                interval = l_dict[k][2] - l_dict[k][1]
                            normed_vel = np.abs(float(v) / float(interval))
                            elementwise_vels.append(normed_vel)
                            break
                mean_normed_vel = \
                    sum(elementwise_vels)/float(len(elementwise_vels))
                if mean_normed_vel < 4:  # neglect outliers
                    iter_vels.append(mean_normed_vel)

                # collect positions
                position_dict = dict(configreader['Position'])
                for k, v in position_dict.items():
                    for u_key, l_dict in hyper_intervals.items():
                        if k in l_dict.keys():
                            # check distribution
                            dist = l_dict[k][0]
                            # positions
                            if dist == 'xor':
                                param_dists[k].append(int(float(v)))
                                interval = len(l_dict[k][1:])
                                lb = 0
                            else:
                                assert dist == 'log' or dist == 'uniform'
                                param_dists[k].append(float(v))
                                interval = l_dict[k][2] - l_dict[k][1]
                                lb = l_dict[k][1]
                            # variances
                            if dist == 'xor' or k in ['active', 'n_hl']:
                                pos = int(float(v))
                            else:
                                pos = float(v)
                            if k in ('weightdecay', 'gaussnoise', 'lr_init'):
                                pos = np.log10(pos)
                                interval = np.log10(l_dict[k][2]) - \
                                           np.log10(l_dict[k][1])

                            unit_interval_position = \
                                (pos - lb)/float(interval)
                            iter_vars[k].append(unit_interval_position)
                            break

        vel_per_iter.append(np.asarray(iter_vels))
        eval_per_iter.append(np.asarray(iter_evals))
        print('considered particles for eval: {}'.format(len(iter_evals)))
        for k in var_per_iter:
            var_per_iter[k].append(np.asarray(iter_vars[k]).var())
    assert len(eval_per_iter) == len(iter_paths)
    assert len(vel_per_iter) == len(iter_paths)
    for k in var_per_iter:
        assert len(var_per_iter[k]) == len(iter_paths)
    imagepath, run_specifier = split(resultspath)
    xticks = np.arange(1, len(iter_paths) + 1, 1)

    w = 7
    h = w * (sqrt(5)-1.0)/2.0
    plt.figure('Evaluations and global best trend', figsize=(w, h))
    gbest_per_iter = np.roll(gbest_per_iter, -1)
    gbest_per_iter[-1] = gbest_per_iter[-2]
    plt.boxplot(eval_per_iter)
    plt.plot(xticks, gbest_per_iter, color='green')
    #plt.title('particle test errors and global best trend')
    plt.xticks(xticks[9::10], [str(s) for s in xticks[9::10]])
    plt.xlabel('iteration')
    plt.ylabel('mean MSE in K²')
    plt.ylim(ymax=50)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(join(imagepath, 'Eval_per_Iter_'+run_specifier+'.svg'),
                format='svg')

    w = 10
    h = w * (sqrt(5)-1.0)/2.0 - 1
    plt.figure('Velocities', figsize=(w, h))
    plt.boxplot(vel_per_iter)
    plt.title('velocity distribution per iteration')
    plt.xticks(xticks[4::5], [str(s) for s in xticks[4::5]])
    plt.xlabel('iteration')
    plt.ylabel('mean normalized velocity')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(join(imagepath, 'Vel_per_Iter_'+run_specifier+'.svg'),
                format='svg')

    order = ['weight_init_distribution', 'n_hl',
             'weight_init_scaling', 'active',
             'preprocess', 'arch', 'optimizer',
             'gaussnoise', 'lr_init', 'n_units', 'seq_len', 'lookback',
             'pca_var_ratio_to_keep', 'weightdecay', 'lr_decay']

    plt.figure('Parameter variance per iteration', figsize=(w, h))
    colormap = plt.cm.gist_ncar
    plt.gca().set_prop_cycle(
        cycler('color',
               [colormap(i) for i in
                np.linspace(0, 0.96, 10)]))
    lines = plt.plot(xticks, gbest_per_iter, label='gobal best', lw=4.0)
    plt.xlabel('iterations')
    plt.ylabel('global best mean MSE in K²')
    plt.twinx()
    styles = ['solid', ] * 7 + ['dashed', ] * 7 + ['dotted', ]

    for i, k in enumerate(order):
        lines += plt.plot(xticks, var_per_iter[k], label=k, ls=styles[i],
                          lw=2.0)
    plt.ylabel('unit interval variance')
    plt.ylim(ymax=0.1)
    plt.legend(lines, [l.get_label() for l in lines],
               ncol=4, loc='upper center', bbox_to_anchor=(0., 1.25, 1., .102),
               columnspacing=1.0, labelspacing=0.0, handletextpad=0.0,
               handlelength=1.5)

    order = ['weight_init_distribution', 'n_hl',  'seq_len',
             'weight_init_scaling', 'active', 'n_units',
             'preprocess', 'lookback', 'pca_var_ratio_to_keep',
             'arch', 'weightdecay', 'lr_decay',
             'optimizer', 'gaussnoise', 'lr_init']

    plt.figure('Parameter Distribution', figsize=(w, 15))
    b = 0
    for par_k in order:
        par_list = param_dists[par_k]
        b += 1
        plt.subplot(5, 3, b)
        if par_list[0] == 'xor':
            ticks, counts = np.unique(np.asarray(par_list[1:]),
                                      return_counts=True)
            x_labels = [v[par_k][1:] for (k, v) in hyper_intervals.items() if
                        par_k in v]
            plt.bar(ticks, counts, align='center', tick_label=x_labels[0])
            plt.ylabel('number of models')
        else:
            assert par_list[0] == 'log' or par_list[0] == 'uniform'
            # special case
            if par_k in ('n_hl', 'active'):
                data = [int(x) for x in par_list[1:]]
                ticks, counts = np.unique(np.asarray(data), return_counts=True)
                plt.bar(ticks, counts, align='center',
                        tick_label=[str(x) for x in list(ticks)])
                plt.ylabel('number of models')
            else:
                plt.boxplot(np.asarray(par_list[1:]), vert=False)
                if b > 12:
                    plt.xlabel('interval')
                plt.yticks([1, ], ['', ])
                if par_k in ('gaussnoise', 'weightdecay', 'lr_init'):
                    plt.xscale('log')
                if par_k in ('seq_len', 'lookback'):
                    plt.xticks(rotation=27)
        plt.title(par_k)
        plt.tight_layout()

    plt.savefig(join(imagepath, 'Param_dist_'+run_specifier+'.svg'),
                format='svg')

    plt.show()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run PSO Plots')
    parser.add_argument('--root', '-r',
                        help='Path to iteration directories')
    parser.add_argument('--collect', action='store_true',
                        help='Flag for collecting results')
    parser.add_argument('--convergence', action='store_true',
                        help='Show velocity per iteration and current '
                             'performance per iteration')
    parser.add_argument('--param_set2', action='store_true',
                        help='Flag for using second adapted hyper parameter '
                             'interval sets')
    args = parser.parse_args()
    matplotlib.rcParams['svg.fonttype'] = 'none'  # for latex pics

    hyperset = 2 if args.param_set2 else 1
    hyperset = 2 if 'Run3' in split(args.root)[1] else hyperset
    if args.collect:
        collect_results(args.root)
    if args.convergence:
        plot_convergence(args.root, hyper=hyperset)
