import tensorflow as tf
keras = tf.keras

train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

model = keras.Sequential()
model.add(keras.Conv2D(32, (3, 3), padding='valid', activation='relu', input_shape=(28, 28, 1)))
model.add(keras.Conv2D(32, (3, 3), padding='valid', activation='relu'))
model.add(keras.Conv2D(32, (3, 3), padding='valid', activation='relu'))
model.add(keras.Conv2D(32, (3, 3), padding='valid', activation='relu'))
model.add(keras.MaxPooling2D((2, 1)))
model.add(keras.Dropout(0.2))
model.add(keras.Conv2D(64, (3, 3), activation='relu'))
model.add(keras.Conv2D(64, (3, 3), activation='relu'))
model.add(keras.MaxPooling2D((2, 1)))
model.add(keras.Dropout(0.2))
model.add(keras.Conv2D(128, (12, 9), activation='relu'))
model.add(keras.Conv2D(25, (1, 1), activation='linear'))
model.add(keras.AveragePooling2D((13, 3), activation='linear'))
model.add(keras.Softmax)
model.summary()

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train  #batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val # batch_size
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


def conv(network, batch_norm, num_layers, num_filters, filter_size, pad,
         pool_size, dropout):
    for k in range(num_layers):
        network = lnn.layers.Conv2DLayer(
            network, num_filters=num_filters,
            filter_size=filter_size,
            W=lnn.init.Orthogonal(gain=np.sqrt(2 / (1 + .1 ** 2))),
            pad=pad,
            nonlinearity=lnn.nonlinearities.rectify,
            name='Conv_{}'.format(k))
        if batch_norm:
            network = lnn.layers.batch_norm(network)

    if pool_size:
        network = lnn.layers.MaxPool2DLayer(network, pool_size=pool_size,
                                            name='Pool')
    if dropout > 0.0:
        network = lnn.layers.DropoutLayer(network, p=dropout)

    return network