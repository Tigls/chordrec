# %%
import os
import pickle

import yaml
from termcolor import colored
from chordrec import data
from chordrec.helpers import dmgr
from chordrec import features
from chordrec import targets
from chordrec.experiment import TempDir, setup, compute_features
from tensorflow import keras
import matplotlib.pyplot as plt

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
#%%
batch_size = 512

train_iterator = dmgr.generator.iterate_aggregated_data(train_set, batch_size)
val_iterator = dmgr.generator.iterate_aggregated_data(val_set, batch_size)
test_iterator = dmgr.generator.iterate_aggregated_data(test_set, 1)

checkpoint = keras.callbacks.ModelCheckpoint('.\checkpoints\chordrec_{epoch}.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='max')
early = keras.callbacks.EarlyStopping(monitor='loss', min_delta=1e-2, patience=15, verbose=1)
tensorboard_cbk = keras.callbacks.TensorBoard(log_dir='.\logs')
callbacks_list = [checkpoint, early, tensorboard_cbk]

model = keras.Sequential()
model.add(keras.layers.InputLayer(input_shape=(15, 105)))
model.add(keras.layers.Reshape((15, 105, 1)))
model.add(keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(keras.layers.BatchNormalization())
#model.add(keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
#model.add(keras.layers.BatchNormalization())
#model.add(keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
#model.add(keras.layers.BatchNormalization())
#model.add(keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(keras.layers.MaxPooling2D((1, 2)))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(keras.layers.BatchNormalization())
#model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
#model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPooling2D((1, 2))),
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Conv2D(128, (9, 12), activation='relu'))
#model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Conv2D(25, (1, 1), activation='linear'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.GlobalAveragePooling2D())
model.summary()
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

train_steps = 430
val_steps = 73
epochs = 170
# %%
history = model.fit_generator(
    generator=train_iterator,
    steps_per_epoch=train_steps,
    epochs=epochs,
    validation_data=val_iterator,
    validation_steps=val_steps,
    callbacks=callbacks_list
)
# %%
model.save('./model_11_10.h5')
history.save()
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

with open('./models/trainHistoryDict', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

predictions = model.predict(a)