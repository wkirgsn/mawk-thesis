import matplotlib.pyplot as plt
import matplotlib as mlp
import numpy as np
from mawk.datascript import DataPack
import argparse
from os.path import split
import pandas as pd
from visuals import setup_vis_trainer

predictions_store = '/home/wilhelmk/MAWK_trainresults/pso/hdf/predictions'
evaluations_store = '/home/wilhelmk/MAWK_trainresults/pso/hdf/evaluations'

pm_best_path = \
    '/home/wilhelmk/MAWK_trainresults/pso/Run3_pm_only/iter_16/637a/1a7f99f4'
yoke_best_path = \
    '/home/wilhelmk/MAWK_trainresults/pso/Run2_stator/iter_17/ad93/803e3470'
teeth_best_path = \
    '/home/wilhelmk/MAWK_trainresults/pso/Run2_stator/iter_17/ad93/803e3470'
winding_best_path = \
    '/home/wilhelmk/MAWK_trainresults/pso/Run2_stator/iter_63/85f9/7b42cb2a'

# get predictions and groundtruth
targets = [pm_best_path, yoke_best_path, teeth_best_path, winding_best_path]
target_labels = ['pm', 'statorjoch', 'statorzahn', 'statorwicklung']
target_labels_y = {'pm': ['$\\vartheta_{PM}$', 'Permanent Magnets'],
                 'statorjoch': ['$\\vartheta_{SY}$', 'Stator Yoke'],
                 'statorzahn': ['$\\vartheta_{ST}$', 'Stator Teeth'],
                 'statorwicklung': ['$\\vartheta_{SW}$', 'Stator Winding']}
target_predictions = []
target_groundtruths = []
for t in targets:
    backup_available = True
    with pd.HDFStore(predictions_store) as store:
        remainder, trainer_id = split(t)
        run_id = split(split(split(remainder)[0])[0])[1]
        if run_id + '_' + trainer_id not in store:
            backup_available = False
        if run_id + '_groundtruth' not in store:
            backup_available = False

    if backup_available:
        print('load backup of predictions for {}'.format(t))
        with pd.HDFStore(predictions_store) as store:
            target_predictions.append(store[run_id + '_' +
                                      trainer_id].as_matrix().T)
            target_groundtruths.append(
                store[run_id + '_groundtruth'].as_matrix().T)
    else:
        print('no predictions backed up for {}. Start evaluating '
              'model:'.format(t))
        parser = argparse.ArgumentParser()
        args = parser.parse_args()
        args.workingplace = 'tp'
        args.toy = False
        args.root = ''
        args.nohdf = False
        args.out = t
        trainer = setup_vis_trainer(args)
        _, _, gr, pr = trainer.evaluation_loop(trainer.test_pool,
                                               trainer.data.n_test)
        target_predictions.append(pr)
        target_groundtruths.append(gr)

        # save to hdf
        cols = [trainer.data.targets_tup._fields[o]
                for o in range(pr.shape[0])]
        with pd.HDFStore(predictions_store) as store:
            store[run_id+'_'+trainer.id] = pd.DataFrame(pr.T, columns=cols)
            store[run_id+'_groundtruth'] = pd.DataFrame(gr.T, columns=cols)

assert len(target_groundtruths) == len(target_labels) and len(
    target_predictions) == len(target_labels)

# get drehzahl, drehmoment, ambient und coolant vorlauf
datapack = DataPack(root='tp', xp=np)
x_test, y_test = datapack.load_profiles([20, ])  # load testset
motorspeed = [datapack._convert_namedtup2matrix(x_test[0].drehzahl),
              'Motor Speed', '$n_{mech}$ in 1/min']
torque = [datapack._convert_namedtup2matrix(x_test[0].T_ist),
          'Torque', '$T_x$ in Nm']
ambient = [datapack._convert_namedtup2matrix(x_test[0].ambient),
           'Ambient Temperature', '$\\vartheta_a$ in °C']
coolant = [datapack._convert_namedtup2matrix(x_test[0].vorlauf),
           'Coolant Temperature', '$\\vartheta_c$ in °C']

# prune uninteresting targets in stator prediction away
target_predictions = [target_predictions[0].ravel(),
                      target_predictions[1][0, :],
                      target_predictions[2][1, :],
                      target_predictions[3][2, :]]
target_groundtruths = [target_groundtruths[0].ravel(),
                      target_groundtruths[1][0, :],
                      target_groundtruths[2][1, :],
                      target_groundtruths[3][2, :]]

# Plot everything
mlp.rcParams.update({'font.family': 'serif',
                     'font.size': 7})
plt.figure('best predictions', figsize=(7, 7), dpi=100)
time = np.arange(ambient[0].shape[1], dtype=np.float32)
time /= (2*60)
for i, (pred, truth) in enumerate(zip(target_predictions, target_groundtruths)):
    title = target_labels_y[target_labels[i]][1]
    plt.subplot(6, 2, 1+2*i)
    plt.plot(time, truth)
    plt.plot(time, pred)
    plt.ylabel(target_labels_y[target_labels[i]][0] + ' in °C')
    plt.xlim(xmax=max(time))
    start, end = plt.ylim()
    if title in ('Permanent Magnets', 'Stator Yoke'):
        plt.ylim(ymax=end+3)
        plt.yticks(np.arange(start, end+3, 20))
    if title == 'Stator Winding':
        plt.yticks(np.arange(start, end, 25))
    plt.title(title)
    plt.grid(True)
    # residuals
    plt.subplot(6, 2, 2 + 2*i)
    plt.plot(time, pred-truth, color='red')
    plt.ylabel('$\Delta$'+target_labels_y[target_labels[i]][0] + ' in °C')
    plt.xlim(xmax=max(time))
    start, end = plt.ylim()
    if title == 'Stator Yoke':
        plt.yticks(np.arange(start, end+1, 3))
    if title == 'Stator Teeth':
        plt.yticks(np.arange(start, end, 4))
    if title == 'Stator Winding':
        plt.yticks(np.arange(start, end + 4, 10))
    if title == 'Permanent Magnets':
        plt.yticks(np.arange(start - 1, end, 8))

    plt.title('Deviation: ' + target_labels_y[target_labels[i]][1])
    plt.grid(True)
    print('max. dev for {}: {} °C'.format(target_labels_y[target_labels[i]][1],
                                          np.abs(pred-truth).max()))

i = 8
for m, m_title, m_ylb in [motorspeed, torque, ambient, coolant]:
    i += 1
    plt.subplot(6, 2, i)
    plt.plot(time, m.ravel())
    plt.ylabel(m_ylb)
    plt.title(m_title)
    plt.grid(True)
    plt.xlim(xmax=max(time))
    if m_title in ('Motor Speed', 'Torque'):
        plt.ylim(ymin=0)
    start, end = plt.ylim()
    if m_title == 'Motor Speed':
        plt.yticks(np.arange(start, end+500, 1500))
    if m_title == 'Ambient Temperature':
        plt.yticks(np.arange(start, end, 4))
    if m_title == 'Coolant Temperature':
        plt.yticks(np.arange(start, end + 2, 15))
    if i > 10:
        plt.xlabel('time in minutes')
plt.tight_layout()
plt.show()



