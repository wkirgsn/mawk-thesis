#!/usr/bin/env python

import numpy as np
import argparse
import time
import sys
from visuals import plot_trend
from mawk.trainer import add_args
from mawk.trainer import Trainer

_descr = """MAWK Training script.
"""

# main
if __name__ == "__main__":
    # Collect arguments
    parser = argparse.ArgumentParser(description=_descr)
    add_args(parser)
    args = parser.parse_args()

    # Setup trainer
    trainer = Trainer(args)
    nan_occured = False

    # Learning loop
    start_time = time.time()
    for epoch in range(trainer.n_epochs):
        print('Epoch {}'.format(epoch + 1))
        epoch_start_time = time.time()

        # training
        mean_loss = trainer.training_loop(quiet=True)
        epoch_duration = (time.time() - epoch_start_time)/60.0
        trainer.recorder[0, epoch] = mean_loss
        trainer.recorder[2, epoch] = (time.time() - start_time)/(60.0*60.0)
        print('[train] mean loss (normalized) = {}\t'
              'taken time: {:4.1f} min'.format(mean_loss, epoch_duration))

        # Catch nan
        if np.isnan(mean_loss):
            nan_occured = True

        # validation
        if not args.no_val and not nan_occured:
            mean_loss, real_loss, _, _ = \
                trainer.evaluation_loop(trainer.val_pool, trainer.data.n_val,
                                        quiet=True)
            if np.isnan(mean_loss):
                nan_occured = True

        if nan_occured:
            mean_loss = 1e12
        else:
            print('[val] mean loss (normalized) = {} | '
                  '(real) = {:.4f} K^2'.format(mean_loss,
                                               np.vstack(real_loss).mean()))

        # Save better net and training trend
        trainer.recorder[1, epoch] = mean_loss
        if mean_loss < trainer.best_net['loss']:
            trainer.best_net['net'] = trainer.model
            trainer.best_net['state'] = trainer.optimizer
            trainer.best_net['loss'] = mean_loss
            trainer.save_to_file(serialize=True)
        else:
            trainer.save_to_file()

        duration = (time.time() - start_time)/(60.0*60.0)
        sys.stdout.write('Duration since start: {:3.2f} h | {}\n'.format(
            duration, trainer.id))

        # adjust learning rate if necessary
        t0 = 4
        if epoch > t0:
            if all((trainer.recorder[1, epoch - x] >
                    trainer.recorder[1, epoch - 1 - x]
                    for x in range(1))):
                trainer.adjust_lr(decrease=True)
            elif all((trainer.recorder[1, epoch - x] <
                      trainer.recorder[1, epoch - 1 - x]
                      for x in range(4))):
                trainer.adjust_lr(decrease=False)

        # check for early stopping
        t_earlystop = round((epoch + 1) / 2)
        if epoch + 1 >= trainer.overfit:
            val_mean_losses = np.trim_zeros(trainer.recorder[1, :], trim='b')
            if np.argmin(val_mean_losses) < t_earlystop:
                print('Best val mean loss has been reached within first {} '
                      'epochs already! Early Stop!\n'.format(t_earlystop))
                break
            elif np.std(val_mean_losses[-20:]) < 5e-4:
                print('Last 20 validation performances have a std lower 5e-4.'
                      ' Training converged! Early stop!')
                break

        # break if nan occured
        if nan_occured:
            print('Nan occured, stop training')
            break

        # learning loop end
        print("")

    print('best_net val mean loss: {}'.format(trainer.best_net['loss']))

    if args.plot:
        import matplotlib.pyplot as plt
        plot_trend(trainer)
        plt.show()
