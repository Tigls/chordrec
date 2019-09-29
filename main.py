#%%
import os
import yaml
from termcolor import colored
from chordrec import data
from chordrec.helpers import dmgr
from chordrec import features
from chordrec import targets
from chordrec.experiment import TempDir, setup, compute_features

datasource = {
  'cached': 'true',
  'context_size': 7,
  'datasets': ['beatles'],
  'preprocessors': [],
  'test_fold': [0, 1, 2, 3, 4, 5, 6, 7],
  'val_fold': None,
}
feature_extractor = {
    'name': 'LogFiltSpec',
    'params': {
        'fmax': 2100,
        'fmin': 65,
        'fps': 10,
        'frame_sizes': [8192],
        'num_bands': 24,
        'unique_filters': 'true'
    }
}
target =  {
    'name': 'ChordsMajMin',
    'params': {}
}
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
    print(colored('Need same number of validation and test folds', 'red'))


for test_fold, val_fold in zip(datasource['test_fold'],
                               datasource['val_fold']):

    print(colored('=' * 20 + ' FOLD {} '.format(test_fold) + '=' * 20, 'yellow'))
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

    print(colored('Train Set:', 'blue'))
    print('\t', train_set)
    print(colored('Validation Set:', 'blue'))
    print('\t', val_set)
    print(colored('Test Set:', 'blue'))
    print('\t', test_set)
    print('')