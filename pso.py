"""Particle Swarm Optimization.
Runs scripts on Oculus Cluster dynamically via CCS allocation and adapts new
parameter settings according to the particle swarm mechanism.
"""

import subprocess as sp
from subprocess import call
from os import listdir, makedirs
from os.path import join, split, isfile, isdir
import argparse
import configparser
import numpy as np
import uuid
import random
import time
from mawk import pc2_utils


class Particle(object):
    """Particle class"""
    def __init__(self, **kwargs):
        # optional
        self.id = kwargs.get('id', None)
        self.position = kwargs.get('position', {})
        self.fitness = kwargs.get('fitness', None)
        self.velocity = kwargs.get('velocity', {})
        self.personal_best = kwargs.get('personal_best', {})
        self.personal_best_fitness = kwargs.get('personal_best_fitness', None)
        self.global_best = kwargs.get('global_best', {})
        self.global_best_fitness = kwargs.get('global_best_fitness', None)
        self.settings_path = kwargs.get('settings_path', None)
        self.trainer_dir = kwargs.get('trainer_dir', None)
        self.default_params = kwargs.get('default_params', None)
        self.hyperparamset = hyper_param_intervals_2 if kwargs.get(
            'hyperparamset', None) == 2 else hyper_param_intervals

        self.best_fitness = kwargs.get('best_fitness', None)
        # mandatory
        self.pyscripts = kwargs['pyscripts']
        pso_path = kwargs['iter_path']

        self.has_info = False
        self.trainer_is_evaluated = False

        if self.id is None:
            # generate ID
            while True:
                self.id = str(uuid.uuid4()).split(sep='-')[0][:4]  # 4 digits
                # Set result out path
                self.path = join(pso_path, self.id)
                if isdir(self.path):
                    # ID exists already, generate a new one
                    pass
                else:
                    makedirs(self.path)
                    break
        else:
            self.path = join(pso_path, self.id)

        # Generate settings file
        self.cur_config = configparser.ConfigParser()
        if self.settings_path is None:
            self.initialize_with_random_params()
            self.write_settings()
        else:
            self.read_settings()

        if isfile(join(self.path, 'fitness.ini')):
            self.read_fitness()

    @staticmethod
    def pick_random_param(dist_id, *kargs):
        """Pick a parameter value within the given intervals

        Args:
            dist_id (String): Distribution. Either 'xor', 'uniform' or 'log'.
            *args: Either strings to pick randomly from in combination with
                dist_id='xor', or two values, min and max, denoting the
                interval limits when dist_id='log' or dist='uniform'.

        Returns (tuple): The Parameter for the settings file, pos and vel

        """
        if dist_id == 'xor':
            # return random.choice(kargs)
            n_args = len(kargs)
            pos = random.uniform(0, n_args)
            vel = random.uniform(-n_args/2, n_args/2)
        elif dist_id == 'uniform':
            if type(kargs[0]) == int:
                pos = random.uniform(kargs[0], kargs[1] + 1)
            else:
                pos = random.uniform(kargs[0], kargs[1])
            vel = random.uniform((kargs[0] - kargs[1])/2,
                                 (kargs[1] - kargs[0])/2)
        elif dist_id == 'log':
            lb = np.log10(kargs[0])
            ub = np.log10(kargs[1])
            pos = 10**(random.uniform(lb, ub))
            quarter = (ub-lb)/4
            vel = 10**(random.uniform(lb+quarter, ub-quarter))
        else:
            raise EnvironmentError('Given dist_id unknown')
        choice = position2param(pos, dist_id, kargs)

        return choice, pos, vel

    def write_settings(self):
        self.settings_path = join(self.path, 'current_settings.ini')
        with open(self.settings_path, 'w+') as settingsfile:
            self.cur_config.write(settingsfile)

    def read_settings(self):
        rc = self.cur_config.read(self.settings_path)
        if len(rc) == 0:
            raise EnvironmentError('Reading config file ' +
                                   self.settings_path + ' failed!')

    def read_fitness(self):
        fitness_conf = configparser.ConfigParser()
        fitness_conf['Fitness'] = {}
        fitness_section = fitness_conf['Fitness']
        ret = fitness_conf.read(join(self.path, 'fitness.ini'))
        if len(ret) > 0:
            if fitness_conf.has_option('Fitness', 'current'):
                self.fitness = float(fitness_section['current'])
            if fitness_conf.has_option('Fitness', 'personal_best'):
                self.personal_best_fitness = float(fitness_section['personal_best'])
            if fitness_conf.has_option('Fitness', 'global_best'):
                self.global_best_fitness = float(fitness_section['global_best'])

    def write_fitness(self):
        fitness_conf = configparser.ConfigParser()
        fitness_conf['Fitness'] = {}
        fitness_section = fitness_conf['Fitness']
        if self.fitness is not None:
            fitness_section['current'] = str(self.fitness)
        if self.personal_best_fitness is not None:
            fitness_section['personal_best'] = str(self.personal_best_fitness)
        if self.global_best_fitness is not None:
            fitness_section['global_best'] = str(self.global_best_fitness)
        with open(join(self.path, 'fitness.ini'), 'w+') as f:
            fitness_conf.write(f)

    def write_info(self):
        info_config = configparser.ConfigParser()
        for f in listdir(self.trainer_dir):
            if f.startswith('evaluation_result'):
                info_config.read(join(self.trainer_dir, f))

        self.fitness = float(info_config['EvaluationResults']
                             ['test_mean_loss_real'])
        # When initialized, pbest and gbest are None
        if self.personal_best_fitness is None and self.global_best_fitness is\
                None:
            assert self.personal_best == self.position
            assert self.global_best == self.position
            self.personal_best_fitness = self.fitness
            self.global_best_fitness = self.fitness

        self.write_fitness()  # overwrite current fitness from last iteration

        info_config['DEFAULT'] = self.default_params
        info_config['Position'] = self.position
        info_config['Velocity'] = self.velocity
        info_config['Personal_Best'] = self.personal_best
        info_config['Global_Best'] = self.global_best
        info_config['Fitness'] = {'current': str(self.fitness),
                                  'personal_best':
                                      str(self.personal_best_fitness),
                                  'global_best': str(self.global_best_fitness)}
        info_ini = join(self.path, 'particle_info_' + self.id + '.ini')
        with open(info_ini, 'w+') as f:
            info_config.write(f)

    def read_info(self):
        info_config = configparser.ConfigParser()

    def initialize_with_random_params(self):
        # add default
        self.cur_config['DEFAULT'] = self.default_params

        # Select hyper parameters
        for section in self.hyperparamset.keys():
            self.cur_config[section] = {}
            for k, v in self.hyperparamset[section].items():
                self.cur_config[section][k], \
                self.position[k], \
                self.velocity[k] = self.pick_random_param(*v)
        self.personal_best = self.position.copy()
        self.global_best = self.position.copy()

        self.cur_config['Position'] = self.position
        self.cur_config['Velocity'] = self.velocity
        self.cur_config['Personal_Best'] = self.personal_best
        self.cur_config['Global_Best'] = self.global_best

        # unset Regularization if not active
        reg_section = self.cur_config['Regularization']
        regs = [int(p) for p in
                format(int(reg_section['active']), '02b')]
        reg_section.pop('active')
        keys = sorted(['gaussnoise', 'weightdecay'])
        assert len(regs) == len(keys), "{} != {}".format(len(regs), len(keys))
        [reg_section.pop(keys[i]) for i, r in enumerate(regs) if r == 0]

    def reinit(self):
        # clear cur_config
        for section in self.hyperparamset.keys():
            self.cur_config.remove_section(section)
        self.initialize_with_random_params()

    @staticmethod
    def load_particle(particle_dir, pyscripts, from_info=True, hyperparamset=1):
        # read particle evaluation
        iter_path, p_id = split(particle_dir)
        info = configparser.ConfigParser()
        label = '_'.join(['particle', 'info', p_id]) if from_info \
            else 'current_settings'
        rc = info.read(join(particle_dir, label + '.ini'))
        if len(rc) == 0:
            return None

        pos = dict(info['Position'])
        vel = dict(info['Velocity'])
        pbest = dict(info['Personal_Best'])
        gbest = dict(info['Global_Best'])

        # Remove default values
        default_options = info.defaults().copy()
        [default_options.pop(m) for m in ['gaussnoise', 'weightdecay']]
        [pos.pop(m) for m in default_options]
        [vel.pop(m) for m in default_options]
        [pbest.pop(m) for m in default_options]
        [gbest.pop(m) for m in default_options]

        specs = {'id': p_id,
                 'position': pos,
                 'velocity': vel,
                 'personal_best': pbest,
                 'global_best': gbest,
                 'settings_path': join(particle_dir, 'current_settings.ini'),
                 'pyscripts': pyscripts,
                 'iter_path': iter_path,
                 'hyperparamset': hyperparamset,
                 'default_params': info.defaults().copy()
                 }
        return Particle(**specs)

    def start_training(self, dry=False):
        """Create and run train_job_script.

        When script finishes, trainer directory in particle directory has all
        its files completed except for evaluation.ini-file. Trainer is fully
        trained.

        Args:
            dry (Boolean): Create scripts but do not execute.
        """
        train_script = join(self.pyscripts, 'training.py')
        jobname = '_'.join([self.default_params['outmask'], 'mawk', 'train',
                            self.id])
        resources_plan = pc2_utils.calculate_resources(
                int(self.cur_config['Net']['n_hl']),
                int(self.cur_config['Net']['n_units']),
                int(self.cur_config['Training']['seq_len']))
        py_line = ' '.join([train_script, '-wp pc2 -c',
                            self.settings_path, '-o', self.path])
        lines = pc2_utils.build_shell_script_lines(self.path, jobname,
                                                   resources_plan, py_line)
        job_script = join(self.path, 'train_job_' + self.id + '.sh')
        pc2_utils.create_n_run_script(job_script, lines, dry)

    def start_evaluating(self, dry=False):
        """Create and run evaluating-job-script

        Assumes a fully trained trainer directory in particle directory.
        Produces an evaluation.ini-file in the trainer directory.

        Args:
            dry (Boolean): Create scripts but do not execute.
        """
        visualize_script = join(self.pyscripts, 'visuals.py')
        jobname = '_'.join([self.default_params['outmask'], 'mawk',
                            'evaluate', self.id])
        resources_plan = {'duration': '30m',
                          'rset': '1',
                          'ncpus': '2',
                          'mem': '12g',
                          'vmem': '12g'}
        py_line = ' '.join([visualize_script, '-wp pc2 -w -o',
                            self.trainer_dir])
        lines = pc2_utils.build_shell_script_lines(self.path, jobname,
                                                   resources_plan, py_line)
        job_script = join(self.path, 'eval_job_' + self.id + '.sh')
        pc2_utils.create_n_run_script(job_script, lines, dry)

    # todo: are parameters in log domain dist really updating their velocity
    # according to the exponent??
    def update_motion(self, c1, cmax, *informants):
        self.read_fitness()
        assert self.fitness is not None
        assert self.personal_best_fitness is not None
        assert self.global_best_fitness is not None

        informants_best_known_positions = [i.global_best for i in informants]
        informants_best_known_fitness = \
            [i.global_best_fitness for i in informants]

        # personal
        if self.fitness < self.personal_best_fitness:
            self.personal_best_fitness = self.fitness
            self.personal_best = dict(self.position).copy()

        # global
        if any(fit < self.global_best_fitness for fit in
               informants_best_known_fitness):
            self.global_best_fitness = min(informants_best_known_fitness)
            self.global_best = dict(informants_best_known_positions[
                informants_best_known_fitness.index(
                    self.global_best_fitness)]).copy()

        # compare personal/global
        if self.personal_best_fitness < self.global_best_fitness:
            self.global_best_fitness = self.personal_best_fitness
            self.global_best = self.personal_best.copy()

        # update velocity
        for k, v in self.velocity.items():
            self.velocity[k] = \
                str(c1*float(v) +
                random.random()*cmax*(float(self.personal_best[k]) -
                float(self.position[k])) +
                random.random()*cmax*(float(self.global_best[k]) -
                float(self.position[k])))

        # update position with offbounds check
        for k, p in self.position.items():
            temp = float(p) + float(self.velocity[k])
            for sec_k, sec_v in self.hyperparamset.items():
                if k in sec_v:
                    dist = sec_v[k][0]

                    # get bounds
                    lb, ub = get_hyperparam_bounds(dist, sec_v[k][1:])

                    # offbounds check
                    if lb <= temp <= ub:
                        pass
                    else:
                        temp = min(max(temp, lb), ub)
                        self.velocity[k] = '0.0'

                    # transform position into choice
                    self.cur_config[sec_k][k] =\
                        position2param(temp, dist, sec_v[k][1:])
                    break  # Dont check remaining dict entries

            self.position[k] = str(temp)

        self.cur_config['Position'] = self.position
        self.cur_config['Velocity'] = self.velocity
        self.cur_config['Personal_Best'] = self.personal_best
        self.cur_config['Global_Best'] = self.global_best

    def clear_evaluations(self):
        self.trainer_dir = None
        self.trainer_is_evaluated = False
        self.has_info = False
        self.fitness = None

    def has_trainer_evaluated(self):
        if not self.trainer_is_evaluated:
            for f in listdir(self.path):
                if isdir(join(self.path, f)):
                    self.trainer_dir = join(self.path, f)
            if self.trainer_dir is not None:
                self.trainer_is_evaluated = any(f.startswith(
                    'evaluation_result_') for f in
                           listdir(join(self.path, self.trainer_dir)))
        return self.trainer_is_evaluated

    def has_particle_info(self):
        if not self.has_info:
            self.has_info = any(f.startswith('particle_info_') for f in
                               listdir(self.path))
        return self.has_info


class Swarm(object):
    def __init__(self, reinit=False, **kwargs):
        self.size = int(kwargs['size'])
        self.path = kwargs['results']
        self.pyscripts = kwargs['pyscripts']
        self.self_confidence = float(kwargs['self_confidence'])
        self.max_confidence_in_others = float(kwargs['confidence_in_others'])
        self.n_informants = int(kwargs['informants'])
        self.n_iterations = int(kwargs['iterations'])
        self.hyperparamset = kwargs['hyperparamset']

        self.default_params = {'dropout': 'no',
                               'weightdecay': 'no',
                               'gradclipping': '15',
                               'gaussnoise': 'no',

                               'outmask': str(kwargs['outmask']),

                               'batchsize': '128',
                               'epochs': '200',
                               'earliest_early_stop': '50',

                               'lossfunc': 'mse'}

        # Check for existing iterations
        self.last_iter = 0
        if any(x.startswith('iter_') for x in listdir(self.path)):
            # Check which iteration it has to continue from
            l = sorted(listdir(self.path), key=lambda k: int(k.split('_')[-1]))
            if l:
                self.last_iter = int(l[-1].split(sep='_')[-1])

        print('Last found iteration: ' + str(self.last_iter))
        if self.last_iter == 0 or reinit:
            iter_path = 'iter_r_' if args.reinit else 'iter_'
            self.last_iter += 1
            iter_path += str(self.last_iter)
            iter_path = join(self.path, iter_path)
            specs = {'iter_path': iter_path,
                     'pyscripts': self.pyscripts,
                     'default_params': self.default_params,
                     'hyperparamset': self.hyperparamset}
            self.particles = [Particle(**specs) for m in range(self.size)]
        else:
            self.particles = []

    def reinit(self):
        for p in self.particles:
            p.reinit()

    def is_busy(self):
        try:
            info = sp.run(['ccsinfo', '-s', '--mine'], universal_newlines=True,
                          stdout=sp.PIPE).stdout
        except Exception as ex:
            info = None
            print('Error while trying to read ccsinfo: ' + str(ex))
        if info is not None:
            infolines = info.split(sep='\n')
            still_busy = False
            for s in infolines:
                if ('ALLOCATED' in s or 'PLANNED' in s) and \
                            ' ' + self.default_params['outmask'] + '_' in s:
                    still_busy = True
            return still_busy
        else:
            raise EnvironmentError('Could not read ccsinfo. '
                                   'Cannot determine swarm status')

    def load_particles(self):
        incomplete = False
        iter_path = join(self.path, 'iter_' + str(self.last_iter))
        for p_id in listdir(iter_path):
            if isdir(join(iter_path, p_id)):
                particle = Particle.load_particle(join(iter_path, p_id),
                                                  self.pyscripts,
                                                  hyperparamset=
                                                  self.hyperparamset)
                if particle is None:
                    incomplete = True
                    self.particles.append(Particle.load_particle(join(
                        iter_path, p_id), self.pyscripts, from_info=False,
                        hyperparamset=self.hyperparamset))
                else:
                    self.particles.append(particle)
        return incomplete

    def send_forth(self, dry=False):
        for p in self.particles:
            if not p.has_particle_info():
                if p.has_trainer_evaluated():
                        p.write_info()
                elif p.trainer_dir is not None and \
                        any(f.endswith('.model') for f in
                            listdir(p.trainer_dir)):
                        p.start_evaluating(dry)
                else:
                    p.start_training(dry)

    def early_stop_check(self):
        if self.last_iter >= self.n_iterations:
            return True

        # read test mean loss real values and compare with threshold
        status_checker = configparser.ConfigParser()
        early_stop_criterion = 1  # Kelvin squared
        for p in self.particles:
            assert p.has_particle_info()
            f = status_checker.read(join(p.path,
                                         'particle_info_' + p.id + '.ini'))
            assert len(f) > 0
            if float(status_checker['EvaluationResults'][
                         'test_mean_loss_real']) < early_stop_criterion:
                print('Goal reached with ' +
                      status_checker['EvaluationResults'][
                          'test_mean_loss_real'] + ' K² as test mean loss '
                      '(real) at particle ' + p.id + ' with model in ' +
                      p.trainer_dir + ' while iter_' + str(self.last_iter))
                return True

        return False

    def supervise(self, dry=False):
        # wait for completion
        while True:
            if self.is_busy():
                print('sleep 2 min')
                time.sleep(120)  # wait 2 mins
                print('wake up')
            else:
                # Swarm is still
                break

        if len(self.particles) == 0:
            # Load Swarm
            print('load swarm')
            self.load_particles()
            assert len(self.particles) > 0, 'Loading particles failed!'

        # check if every particle has info
        all_finished = all(p.has_particle_info() for p in self.particles)
        if not all_finished:
            self.send_forth(dry)
            return False

        # Swarm is still, particles list in swarm is initialized
        # every particle has created an evaluation
        # start new iteration if goal not reached yet
        if self.early_stop_check():
            return True

        self.last_iter += 1
        print('start new iteration: ' + str(self.last_iter))
        iter_path = join(self.path, 'iter_'+str(self.last_iter))
        makedirs(iter_path)
        for p in self.particles:
            informants = [random.choice(self.particles) for m in
                              range(self.n_informants)]
            p.update_motion(self.self_confidence,
                            self.max_confidence_in_others,
                            *informants)
            p.path = join(iter_path, p.id)
            makedirs(p.path)
            p.write_settings()
            p.write_fitness()
            p.clear_evaluations()
        return False


hyper_param_intervals = \
    {'Net': {'arch': ['xor', 'lstm', 'gru', 'lstm_peep'],
             'n_hl': ['uniform', 1, 5],
             'n_units': ['log', 2, 128],
             'weight_init_distribution': ['xor', 'unit_normal', 'uniform'],
             'weight_init_scaling': ['xor', 'normalized_init', 'standard_init']
             },
     'Training': {'seq_len': ['uniform', 30, 7800],
                  'preprocess': ['xor', 'normalize', 'pca'],
                  'pca_var_ratio_to_keep': ['uniform', 0.5, 1.0],
                  'lookback': ['uniform', 0, 1000]},
     'Opt': {'optimizer': ['xor', 'adam', 'nesterov', 'sgd'],
             'lr_init': ['log', 1e-4, 1e-1],
             'lr_decay': ['uniform', 0.5, 0.99]},
     'Regularization': {'active': ['uniform', 0, 3],
                        'gaussnoise': ['log', 1e-7, 1e-3],
                        'weightdecay': ['log', 1e-7, 1e-4]}
     }

hyper_param_intervals_2 = \
    {'Net': {'arch': ['xor', 'lstm', 'gru', 'lstm_peep'],
             'n_hl': ['uniform', 1, 3],
             'n_units': ['log', 2, 256],
             'weight_init_distribution': ['xor', 'unit_normal', 'uniform'],
             'weight_init_scaling': ['xor', 'normalized_init', 'standard_init']
             },
     'Training': {'seq_len': ['uniform', 30, 7880],
                  'preprocess': ['xor', 'normalize', 'pca'],
                  'pca_var_ratio_to_keep': ['uniform', 0.5, 1.0],
                  'lookback': ['uniform', 0, 2000]},
     'Opt': {'optimizer': ['xor', 'adam', 'nesterov', 'sgd'],
             'lr_init': ['log', 1e-4, 1e-1],
             'lr_decay': ['uniform', 0.5, 0.99]},
     'Regularization': {'active': ['uniform', 0, 3],
                        'gaussnoise': ['log', 1e-7, 1e-3],
                        'weightdecay': ['log', 1e-7, 1e-4]}
     }


def position2param(position, dist_id, *args):
    """Given a floating point position, returns the corresponding choice for
    the pso-config.ini file.

    Args:
        position (float): Particle position
        dist_id (String): Either "xor", "uniform" or "log"
        *args: List of either possible choices (xor) or list of two values;
            lower bound and upper bound.

    Returns: The corresponding choice to write to the config file for pso.

    """
    choices = args[0]
    if dist_id == 'xor':
        choice = choices[int(position)]
    elif dist_id == 'uniform':
        choice = str(int(position)) if type(choices[0]) == int \
            else str(position)
    elif dist_id == 'log':
        choice = str(int(round(position))) if type(choices[0]) == int \
            else str(position)
    assert choice is not None
    return choice


def get_hyperparam_bounds(dist_id, *args):
    """Returns the hyperparameter bounds.

    Args:
        dist_id: Either "xor", "uniform" or "log"
        *args: List of possible choices (xor) or list of two values;
            lower bound and upper bound.

    Returns: lower bound, upper bound as floats

    """
    choices = args[0]
    if dist_id == 'xor':
        lb = 0.0
        ub = float(len(choices)) - 1e-9
    else:
        assert dist_id == 'log' or dist_id == 'uniform'
        lb = choices[0]
        ub = choices[1] + (1-1e-9) if type(lb) == int and dist_id == 'uniform'\
            else choices[1]
    return lb, ub


def num(s):
    try:
        return int(s)
    except ValueError:
        return float(s)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run PSO')
    parser.add_argument('--config', '-c',
                        default=join('settings', 'pso_config.ini'),
                        help='Path to config file')
    parser.add_argument('--reinit', '-r', action='store_true',
                        help='reinitialize the swarm')
    parser.add_argument('--dry', action='store_true',
                        help='Build environment but do not allocate jobs')
    parser.add_argument('--outmask', default=15, type=int,
                        help='Use this integer as binary mask for neglecting '
                             'certain output targets')
    parser.add_argument('--param_set2', action='store_true',
                        help='Flag for using second adapted hyper parameter '
                             'interval sets')
    args = parser.parse_args()

    config = configparser.ConfigParser()
    rc = config.read(args.config)
    if len(rc) == 0:
        raise EnvironmentError('Given config file not found: '
                               '{}'.format(args.config))

    swarmconfig = dict(config['Path'])
    swarmconfig.update(dict(config['Swarm']))
    swarmconfig['outmask'] = args.outmask
    swarmconfig['hyperparamset'] = 2 if args.param_set2 else 1
    swarm = Swarm(args.reinit, **swarmconfig)

    while True:
        goal_reached = swarm.supervise(args.dry)
        if goal_reached:
            print('Swarm reached goal! Finish supervising.')
            break

