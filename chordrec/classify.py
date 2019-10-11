import os

import yaml
from    termcolor import colored

from chordrec import data
from chordrec.helpers import dmgr
from chordrec import features
# import nn
from chordrec import targets
# from . import test

# from chordrec.models import dnn, avg_gap_feature, crf, rnn
from chordrec.experiment import TempDir, setup, compute_features #create_optimiser

# Initialise Sacred experiment
ex = setup('Classify Chords')


# Standard config
# @ex.config
# def _cfg():
#     observations = 'results'
#     feature_extractor = None
#     target = None
#     model = None
#     optimiser = None
#     training = None
#     regularisation = None
#     testing = None
#     augmentation = None


# add models
# dnn.add_sacred_config(ex)
# avg_gap_feature.add_sacred_config(ex)
# crf.add_sacred_config(ex)
# rnn.add_sacred_config(ex)


# add general configs
# @ex.named_config
# def learn_rate_schedule():
#     optimiser = dict(
#         schedule=dict(
#             interval=10,
#             factor=0.5
#         )
#     )


# @ex.automain
def main(_log, datasource, feature_extractor, target, model, optimiser,
         training, regularisation, augmentation, testing):

    err = False
    if model is None or not model or 'type' not in model:
        _log.error(colored('Specify a model!', 'red'))
        err = True
    if feature_extractor is None:
        _log.error(colored('Specify a feature extractor!', 'red'))
        err = True
    if target is None:
        _log.error(colored('Specify a target!', 'red'))
        err = True
    if err:
        return 1

    target_computer = targets.create_target(
        feature_extractor['params']['fps'],
        target
    )

    if not isinstance(datasource['test_fold'], list):
        datasource['test_fold'] = [datasource['test_fold']]

    if not isinstance(datasource['val_fold'], list):
        datasource['val_fold'] = [datasource['val_fold']]

        # if no validation folds are specified, always use the
        # 'None' and determine validation fold automatically
        if datasource['val_fold'][0] is None:
            datasource['val_fold'] *= len(datasource['test_fold'])

    if len(datasource['test_fold']) != len(datasource['val_fold']):
        _log.error(colored('Need same number of validation and test folds', 'red'))
        return 1

    all_pred_files = []
    all_gt_files = []

    print(colored('\nStarting experiment ' + ex.observers[0].hash(), 'magenta'))

    with TempDir() as exp_dir:
        for test_fold, val_fold in zip(datasource['test_fold'],
                                       datasource['val_fold']):
            print('')
            print(colored(
                '=' * 20 + ' FOLD {} '.format(test_fold) + '=' * 20, 'yellow'))
            # Load data sets
            print(colored('\nLoading data...\n', 'red'))

            train_set, val_set, test_set, gt_files = data.create_datasources(
                dataset_names=datasource['datasets'],
                preprocessors=datasource['preprocessors'],
                compute_features=features.create_extractor(feature_extractor,
                                                           test_fold),
                compute_targets=target_computer,
                context_size=datasource['context_size'],
                test_fold=test_fold,
                val_fold=val_fold,
                cached=datasource['cached'],
            )

            if testing['test_on_val']:
                test_set = val_set

            print(colored('Train Set:', 'blue'))
            print('\t', train_set)
            print(colored('Validation Set:', 'blue'))
            print('\t', val_set)
            print(colored('Test Set:', 'blue'))
            print('\t', test_set)
            print('')

            # build network
            print(colored('Building network...\n', 'red'))

            model_type = globals()[model['type']]
            mdl = model_type.build_model(in_shape=train_set.dshape,
                                         out_size=train_set.tshape[0],
                                         model=model)

            # mandatory parts of the model
            neural_net = mdl['network']
            input_var = mdl['input_var']
            target_var = mdl['target_var']
            loss_fn = mdl['loss_fn']

            # optional parts
            mask_var = mdl.get('mask_var')
            feature_out = mdl.get('feature_out')

            train_batches, validation_batches = model_type.create_iterators(
                train_set, val_set, training, augmentation
            )

            opt, lrs = create_optimiser(optimiser)

            train_fn = nn.compile_train_fn(
                neural_net, input_var, target_var,
                loss_fn=loss_fn, opt_fn=opt, mask_var=mask_var,
                **regularisation
            )

            test_fn = nn.compile_test_func(
                neural_net, input_var, target_var,
                loss_fn=loss_fn, mask_var=mask_var,
                **regularisation
            )

            process_fn = nn.compile_process_func(
                neural_net, input_var, mask_var=mask_var)

            if feature_out is not None:
                feature_fn = nn.compile_process_func(
                    feature_out, input_var, mask_var=mask_var
                )
            else:
                feature_fn = None

            print(colored('Neural Network:', 'red'))
            print(nn.to_string(neural_net))
            print('')

            if 'param_file' in training:
                nn.load_params(neural_net,
                               training['param_file'].format(test_fold))
                train_losses = []
                val_losses = []
                val_accs = []
            else:
                if 'init_file' in training:
                    print('initialising')
                    nn.load_params(neural_net,
                                   training['init_file'].format(test_fold))
                print(colored('Starting training...\n', 'red'))
                train_losses, val_losses, _, val_accs = nn.train(
                    network=neural_net,
                    train_fn=train_fn, train_batches=train_batches,
                    test_fn=test_fn, validation_batches=validation_batches,
                    threads=10, callbacks=[lrs] if lrs else [],
                    num_epochs=training['num_epochs'],
                    early_stop=training['early_stop'],
                    early_stop_acc=training['early_stop_acc']
                )
                param_file = os.path.join(
                    exp_dir, 'params_fold_{}.pkl'.format(test_fold))
                nn.save_params(neural_net, param_file)
                ex.add_artifact(param_file)

            print(colored('\nStarting testing...\n', 'red'))
    #
    #         if feature_fn is not None:
    #             dest_dir = os.path.join(exp_dir,
    #                                     'features_fold_{}'.format(test_fold))
    #             compute_features(
    #                 feature_fn, train_set, batch_size=testing['batch_size'],
    #                 dest_dir=dest_dir, extension='.features.npy',
    #                 use_mask=mask_var is not None)
    #             compute_features(
    #                 feature_fn, val_set, batch_size=testing['batch_size'],
    #                 dest_dir=dest_dir, extension='.features.npy',
    #                 use_mask=mask_var is not None)
    #             compute_features(
    #                 feature_fn, test_set, batch_size=testing['batch_size'],
    #                 dest_dir=dest_dir, extension='.features.npy',
    #                 use_mask=mask_var is not None)
    #             ex.add_artifact(dest_dir)
    #
    #         pred_files = test.compute_labeling(
    #             process_fn, target_computer, test_set, dest_dir=exp_dir,
    #             use_mask=mask_var is not None, batch_size=testing['batch_size']
    #         )
    #
    #         test_gt_files = dmgr.files.match_files(
    #             pred_files, test.PREDICTION_EXT, gt_files, data.GT_EXT
    #         )
    #
    #         all_pred_files += pred_files
    #         all_gt_files += test_gt_files
    #
    #         print(colored('Results:', 'blue'))
    #         scores = test.compute_average_scores(test_gt_files, pred_files)
    #         test.print_scores(scores)
    #         result_file = os.path.join(
    #             exp_dir, 'results_fold_{}.yaml'.format(test_fold))
    #         yaml.dump(dict(scores=scores,
    #                        train_losses=list(map(float, train_losses)),
    #                        val_losses=list(map(float, val_losses)),
    #                        val_accs=list(map(float, val_accs))),
    #                   open(result_file, 'w'))
    #         ex.add_artifact(result_file)
    #
    #         # delete datasets so disk space is free
    #         del train_set
    #         del val_set
    #         del test_set
    #
    #     # if there is something to aggregate
    #     if len(datasource['test_fold']) > 1:
    #         print(colored('\nAggregated Results:\n', 'yellow'))
    #         scores = test.compute_average_scores(all_gt_files, all_pred_files)
    #         test.print_scores(scores)
    #         result_file = os.path.join(exp_dir, 'results.yaml')
    #         yaml.dump(dict(scores=scores), open(result_file, 'w'))
    #         ex.add_artifact(result_file)
    #
    #     for pf in all_pred_files:
    #         ex.add_artifact(pf)
    #
    # print(colored('Stopping experiment ' + ex.observers[0].hash(), 'magenta'))
