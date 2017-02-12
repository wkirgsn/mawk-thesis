import pandas as pd
from os import listdir, makedirs, removedirs, rename
import configparser
import argparse
from os.path import join, split, isfile, isdir
import numpy as np
from mawk import pc2_utils
import matplotlib.pyplot as plt
from pso import hyper_param_intervals
from math import sqrt

hyper_bounds = {'arch': ['lstm', 'lstm_peep', 'gru'],
                'n_hl': [1, 5],
                'n_units': [2, 300],
                'weight_init_distribution': ['unit_normal', 'uniform'],
                'weight_init_scaling': ['normalized_init', 'standard_init'],
                'seq_len': [30, 7880],
                'preprocess': ['normalize', 'pca'],
                'pca_var_ratio_to_keep': [0.5, 1.0],
                'lookback': [0, 2000],
                'optimizer': ['adam', 'nesterov', 'sgd'],
                'lr_init': [1e-4, 1e-1],
                'lr_decay': [0.5, 0.99],
                'active': [0, 3],
                'gaussnoise': [1e-7, 1e-3],
                'weightdecay': [1e-7, 1e-4]
                }
hyper_order = ['arch', 'n_hl', 'n_units', 'weight_init_distribution',
               'weight_init_scaling', 'seq_len', 'preprocess', 'lookback',
               'optimizer', 'lr_init', 'lr_decay', 'active']
log_hypers = ['n_units', 'lr_init', 'gaussnoise', 'weightdecay']


def submit_training_job(configuration, pyscripts_path, run_path,
                        jobname, dry=False):
    if not isdir(run_path):
        makedirs(run_path)
    else:
        # check if trainer already trained
        for f in listdir(run_path):
            if isdir(join(run_path, f)) and len(f) == 8:
                if any(x.endswith('.model') for x in listdir(join(run_path,
                                                                  f))):
                    # model already trained
                    return None
                else:
                    # trainer failed, delete it
                    removedirs(join(run_path, f))

    # generate settings.ini
    model_settings = join(run_path, 'settings.ini')
    with open(model_settings, 'w+') as f:
        configuration.write(f)

    resources_plan = pc2_utils.calculate_resources(
        n_hl=int(configuration['Net']['n_hl']),
        n_units=int(configuration['Net']['n_units']),
        seq_len=int(configuration['Training']['seq_len']))
    py_line = ' '.join([join(pyscripts_path, 'training.py'),
                        '-wp pc2 -c',
                        model_settings,
                        '-o', run_path])
    lines = pc2_utils.build_shell_script_lines(run_path,
                                               jobname,
                                               resources_plan,
                                               py_line)
    job_script = join(run_path, 'train_job.sh')
    pc2_utils.create_n_run_script(job_script, lines, dry)


def submit_evaluation_job(pyscripts_path, run_path,
                          jobname, dry=False):

    resources_plan = {'duration': '30m',
                          'rset': '1',
                          'ncpus': '2',
                          'mem': '12g',
                          'vmem': '12g'}

    # get model directory
    model_dir = ''
    for f in listdir(run_path):
        if isdir(join(run_path, f)) and len(f) == 8:
            model_dir = join(run_path, f)
            assert any(x.endswith('.model') for x in listdir(model_dir)), \
                '{} is missing model and state'.format(model_dir)
    assert model_dir != '', 'Could not find model dir in' \
                            '{}'.format(run_path)

    # check if trainer already evaluated
    if any(x.startswith('evaluation_result_') for x in listdir(model_dir)):
        # trainer has been evaluated already
        return None

    py_line = ' '.join([join(pyscripts_path, 'visuals.py'),
                        '-wp pc2 -w -o',
                        model_dir])
    lines = pc2_utils.build_shell_script_lines(run_path,
                                               jobname,
                                               resources_plan,
                                               py_line)
    job_script = join(run_path, 'eval_job.sh')
    pc2_utils.create_n_run_script(job_script, lines, dry)


def scrape_val_test_evaluations(path_to_models):
    # read results of the optimum hyperparam set
    evaluations = []
    runs = listdir(path_to_models)
    for run in runs:
        # get model directory
        model_dir = ''
        for f in listdir(join(path_to_models, run)):
            ff = join(path_to_models, run, f)
            if isdir(ff) and len(f) == 8:
                model_dir = ff
                break
        assert model_dir != '', 'Could not find model dir in' \
                                '{}'.format(run)
        # get evaluation file
        evaluation_file = ''
        for f in listdir(model_dir):
            ff = join(model_dir, f)
            if f.startswith('evaluation_result_'):
                evaluation_file = ff
                break
        assert evaluation_file != '', 'Could not find evaluationfile ' \
                                      'in {}'.format(run)
        # read evaluation file
        eval_config = configparser.ConfigParser()
        rc = eval_config.read(evaluation_file)
        assert len(rc) != 0, 'Given config file not found: {}'.format(
            evaluation_file)
        results = eval_config['EvaluationResults']
        evaluations.append([float(results['val_mean_loss_real']),
                            float(results['test_mean_loss_real'])])
    val = np.atleast_2d([x[0] for x in evaluations]).T
    test = np.atleast_2d([x[1] for x in evaluations]).T
    return pd.DataFrame(np.hstack([val, test]), columns=['val', 'test'])


def start_consistency_training(n, path_to_models, opt_model_config,
                               pyscripts_path, dry=False):
    for i in range(n):
        jobname = 'consistency_run_' + str(i)
        run_path = join(path_to_models, jobname)
        submit_training_job(opt_model_config, pyscripts_path, run_path,
                            jobname, dry)


def start_consistency_evaluation(path_to_models, pyscripts_path, dry=False):
    runs = listdir(path_to_models)
    for run in runs:
        jobname = 'evaluate_'+run
        run_path = join(path_to_models, run)
        submit_evaluation_job(pyscripts_path, run_path, jobname, dry)


def plot_consistency_runs(path_to_models, title='consistency'):

    df = scrape_val_test_evaluations(path_to_models)

    # remove outliers
    std = df['test'].std()
    mu = df['test'].mean()
    old_size = df.shape[0]
    df = df[abs(df['test'] - mu) < 2.5 * std]
    print(df.describe())
    print('removed {} outliers, old std = {:.2f} K²'.format(old_size -
                                                          df.shape[0], std))
    w = 3
    h = 3
    plt.figure(figsize=(w, h))
    plt.hist([df['val'], df['test']], 30,
             histtype='step', label=['val', 'test'])
    plt.grid(True)
    plt.ylabel('number of models')
    #plt.xticks(np.arange(df.shape[1]) + 1, df.columns)
    plt.xlabel('mean MSE in K²')
    plt.title('scatter, {} samples'.format(df.shape[0]))
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    bbox_prop = dict(boxstyle='round,pad=0.5', fc='white')
    """plt.text(xmin + 0.1*(xmax-xmin), ymax - 0.1*(ymax-ymin),
             'val: $\mu={:.1f} K^2$, $\sigma={:.1f} K^2$'.format(
                 df['val'].mean(), df['val'].std()), color='blue',
             bbox=bbox_prop)
    plt.text(xmin + 0.1*(xmax-xmin), ymax - 0.3*(ymax-ymin),
             'test: $\mu={:.1f} K^2$, $\sigma={:.1f} K^2$'.format(
                 df['test'].mean(), df['test'].std()), color='green',
             bbox=bbox_prop)"""
    plt.legend()
    plt.show()


def start_hypervar_training(n, path_to_models, opt_model_config,
                            pyscripts_path, dry=False):

    for hyper in hyper_order:
        hyper_run_path = join(path_to_models, hyper)
        category = [k for (k, v) in dict(hyper_param_intervals).items() if
                        hyper in v]
        assert len(category) == 1
        category = category[0]
        optimum = opt_model_config[category][hyper]

        # special case PCA
        if hyper == 'preprocess':
            if optimum == 'pca':
                opt_model_config[category][hyper] = 'normalize'
                for i in range(n):
                    jobname = 'hypervar_run_' + str(i) + '_' + hyper + \
                              '_normalize'
                    run_path = join(hyper_run_path, jobname)
                    submit_training_job(opt_model_config, pyscripts_path,
                                        run_path, jobname, dry)
                opt_model_config[category][hyper] = optimum
            else:  # optimum == normalize
                opt_model_config[category][hyper] = 'pca'
                sublabel = 'pca_var_ratio_to_keep'
                sub_optimum = opt_model_config[category][sublabel]
                alternatives = [str(hyper_bounds[sublabel][0]),
                                str(hyper_bounds[sublabel][1])]
                for alt in alternatives:
                    opt_model_config[category][sublabel] = alt
                    # repeat for consistency
                    for i in range(n):
                        jobname = 'hypervar_run_' + str(i) + '_' + hyper + \
                                  '_' + alt
                        run_path = join(hyper_run_path, jobname)
                        submit_training_job(opt_model_config, pyscripts_path,
                                            run_path, jobname, dry)
                opt_model_config[category][sublabel] = sub_optimum
                opt_model_config[category][hyper] = optimum
            continue

        # Special Case Regularization
        if hyper == 'active':
            old_weightdecay = opt_model_config[category]['weightdecay']
            old_gaussnoise = opt_model_config[category]['gaussnoise']
            if optimum == '0':
                opt_model_config[category]['weightdecay'] = str(1e-6)
                opt_model_config[category]['gaussnoise'] = str(1e-6)
                alternatives = ['1', '2', '3']

            elif optimum == '1':  # weightdecay selected
                opt_model_config[category]['gaussnoise'] = str(1e-6)
                alternatives = ['0', '2', '3']

            elif optimum == '2':  # gaussnoise selected
                opt_model_config[category]['weightdecay'] = str(1e-6)
                alternatives = ['0', '1', '3']
            elif optimum == '3':
                alternatives = ['0', '1', '2']
            else:
                raise NotImplementedError('active > 3 not implemented')
            for alt in alternatives:
                opt_model_config[category][hyper] = alt
                for i in range(n):
                    jobname = 'hypervar_run_' + str(i) + '_' +hyper+'_'+alt
                    run_path = join(hyper_run_path, jobname)
                    submit_training_job(opt_model_config, pyscripts_path,
                                        run_path, jobname, dry)
            opt_model_config[category]['weightdecay'] = old_weightdecay
            opt_model_config[category]['gaussnoise'] = old_gaussnoise
            continue

        # general case
        if type(hyper_bounds[hyper][0]) == str:
            # xor sweep
            options = hyper_bounds[hyper]
            alternatives = [o for o in options if o != optimum]
        else:
            lb = hyper_bounds[hyper][0]
            ub = hyper_bounds[hyper][1]

            if hyper in log_hypers:
                # log sweeping
                tenpercent = 0.1*np.abs(np.log10(ub)-np.log10(lb))
                over_selection = min(float(ub),
                                     10**(np.log10(float(optimum))+tenpercent))
                under_selection = max(float(lb),
                                      10**(np.log10(float(optimum))-tenpercent))
            else:
                # uniform sweeping
                tenpercent = 0.1 * (ub-lb)
                over_selection = min(float(ub), float(optimum) + tenpercent)
                under_selection = max(float(lb), float(optimum) - tenpercent)

            # integer converting
            if type(lb) == int:
                under_selection = int(under_selection)
                over_selection = int(over_selection)
                if under_selection == int(optimum):
                    under_selection -= 1
                if under_selection < lb:
                    under_selection = None
                if over_selection == int(optimum):
                    over_selection += 1
                if over_selection > ub:
                    over_selection = None

            if over_selection is not None:
                over_selection = str(over_selection)
            if under_selection is not None:
                under_selection = str(under_selection)
            alternatives = [over_selection, under_selection]

        for alt in alternatives:
            if alt is None:
                continue
            opt_model_config[category][hyper] = alt
            # repeat for consistency
            for i in range(n):
                jobname = 'hypervar_run_' + str(i) + '_' + hyper + '_' + alt
                run_path = join(hyper_run_path, jobname)
                submit_training_job(opt_model_config, pyscripts_path,
                                    run_path, jobname, dry)
        # turn hyper config back to optimum
        opt_model_config[category][hyper] = optimum


def start_hypervar_evaluation(path_to_hypervars, pyscripts_path, dry=False):
    hypervars = listdir(path_to_hypervars)
    for hyper in hypervars:
        hyper_path = join(path_to_hypervars, hyper)
        runs = listdir(hyper_path)
        for run in runs:
            jobname = 'evaluate_' + run
            run_path = join(hyper_path, run)
            submit_evaluation_job(pyscripts_path, run_path, jobname, dry)


def plot_hypervar_runs(path_to_hypers):
    hypervars = listdir(path_to_hypers)
    for hyper in hypervars:
        hyper_path = join(path_to_hypers, hyper)
        alternatives = list(set([x.split('_')[-1] for x in listdir(
            hyper_path)]))

        for alt in alternatives:
            if not isdir(join(hyper_path, alt)):
                makedirs(join(hyper_path, alt))
            # move alternatives in own dir
            for run in listdir(hyper_path):
                run_specifiers = run.split('_')
                if run_specifiers[-1] == alt and len(run_specifiers) > 1:
                    rename(join(hyper_path, run), join(hyper_path, alt, run))

            # return mean of the sample
            #plot_consistency_runs(join(hyper_path, alt), hyper + '_' + alt)
            df = scrape_val_test_evaluations(join(hyper_path, alt))
            mu = df['test'].mean()
            print('{} with alternative "{}" has mean: {:.2f} K²'.format(hyper,
                                                                    alt, mu))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run sensitivity analysis')
    parser.add_argument('--config', '-c',
                        default=join('settings', 'sensitivity_config.ini'),
                        help='Path to config file')
    parser.add_argument('--dry', action='store_true',
                        help='Build environment but do not allocate jobs')
    parser.add_argument('--param_set2', action='store_true',
                        help='Flag for using second adapted hyper parameter '
                             'interval sets')
    parser.add_argument('--job_type', '-j', choices=['train', 'evaluate'],
                        default=None,
                        help='Whether scripts for training or for evaluating '
                             'shall be submitted to pc2')
    parser.add_argument('--plot', '-p', action='store_true',
                        help='Analyze sensitivity and plot results instead of'
                             ' sending training/evaluation jobs to pc2 cluster')
    parser.add_argument('--consistency', '-s', action='store_true',
                        help='Evaluate the optimum model consistency by '
                             'repeating training of its set as often as '
                             'defined in the settings')
    args = parser.parse_args()
    config = configparser.ConfigParser()
    rc = config.read(args.config)
    assert len(rc) != 0, 'Given config file not found: {}'.format(args.config)

    store_model_path = config['Path']['store_models']
    optimum_model_path = config['Path']['optimum_model']
    pyscripts = config['Path']['pyscripts']
    reps = int(config['General']['consistency_repetitions'])

    if args.plot:
        # just plot results
        if args.consistency:
            plot_consistency_runs(store_model_path,
                                  split(store_model_path)[1] + ' optimum')
        else:
            plot_hypervar_runs(store_model_path)
    else:
        # Submit jobs to pc2
        if not isdir(store_model_path):
            makedirs(store_model_path)
        head, trainer_id = split(optimum_model_path)
        model_config = configparser.ConfigParser()
        rc = model_config.read(join(optimum_model_path,
                                    trainer_id + '.ini'))
        assert len(rc) != 0, 'Model config file could not be ' \
                             'read: {}'.format(join(optimum_model_path,
                                               trainer_id + '.ini'))
        if args.consistency:
            if args.job_type == 'train':
                # Start consistency trainings with optimum model hyperparam set
                start_consistency_training(reps, store_model_path,
                                           model_config, pyscripts,
                                           dry=args.dry)
            elif args.job_type == 'evaluate':
                # start evaluation jobs
                start_consistency_evaluation(store_model_path, pyscripts,
                                             dry=args.dry)
            else:
                raise EnvironmentError('Please specify job_type in arguments')
        else:
            if args.job_type == 'train':
                start_hypervar_training(reps, store_model_path, model_config,
                                        pyscripts, dry=args.dry)
            elif args.job_type == 'evaluate':
                start_hypervar_evaluation(store_model_path, pyscripts,
                                          dry=args.dry)
            else:
                raise EnvironmentError('Please specify job_type in arguments')



