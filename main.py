
import os
import yaml
from termcolor import colored
from chordrec import data
from chordrec.helpers import dmgr
from chordrec import features
from chordrec import targets
from chordrec.experiment import TempDir, setup, compute_features
from tensorflow import keras

datasource = {
  'cached': 'true',
  'context_size': 7,
  'datasets': ['beatles'],
  'preprocessors': [],
  'test_fold': [0], #
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

train_iterator = dmgr.iterators.iterate_aggregated(train_set, 1)
val_iterator = dmgr.iterators.iterate_aggregated(val_set, 1)

model = keras.Sequential()
model.add(keras.layers.InputLayer(input_shape=(15, 105)))
model.add(keras.layers.Reshape((15, 105, 1)))
model.add(keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(keras.layers.Conv2D(32, (3, 3),  padding='same', activation='relu'))
model.add(keras.layers.Conv2D(32, (3, 3),  padding='same', activation='relu'))
model.add(keras.layers.Conv2D(32, (3, 3),  padding='same', activation='relu'))
model.add(keras.layers.MaxPooling2D((1, 2)))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Conv2D(64, (3, 3),  activation='relu'))
model.add(keras.layers.Conv2D(64, (3, 3),  activation='relu'))
model.add(keras.layers.MaxPooling2D((1, 2)))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Conv2D(128, (9, 12),activation='relu'))
model.add(keras.layers.Conv2D(25, (1, 1), activation='linear'))
model.add(keras.layers.GlobalAveragePooling2D())
model.summary()
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

batch_size = 32
epochs = 1

history = model.fit_generator(
    generator = train_iterator,
    steps_per_epoch=batch_size, #batch_size,
    epochs=epochs,
    validation_data=val_iterator,
    validation_steps=batch_size # batch_size
)