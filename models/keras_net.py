"""Keras model definitions"""

# Dependecy imports
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv3D, Dropout, BatchNormalization
from keras.optimizers import adam
from keras import initializers, regularizers
import keras.backend.tensorflow_backend as K
from keras.callbacks import EarlyStopping

# Set Keras TF backend allow_growth not to consume all GPU memory
K_CONFIG = K.tf.ConfigProto()
K_CONFIG.allow_soft_placement = True
K_CONFIG.gpu_options.allow_growth = True # pylint: disable=E1101
K.set_session(K.tf.Session(config=K_CONFIG))

def conv3d_model(nb_classes, params, summary=False):
    """Create keras model.

    dshape: (timeseries, height, width, channels)
    params:
        PARAMS = {
            '__cnn_dropout_1': 0.3, '__cnn_dropout_2': 0.7, '__cnn_dropout_3': 0.2,
            '__cnn_ksize_1': 1, '__cnn_ksize_2': 6, '__cnn_ksize_3': 5,
            '__cnn_neurons_1': 512, '__cnn_neurons_2': 128, '__cnn_neurons_3': 16,
            '__dense_dropout_1': 0.8, '__dense_dropout_2': 0.8, '__dense_dropout_3': 0.8,
            '__dense_neurons_1': 128, '__dense_neurons_2': 8, '__dense_neurons_3': 16
            'activation_cnn': 'relu',
            'activation_dense': 'elu',
            'batch_norm': True,
            'batch_size': 128,
            'class_weight': True,
            'dshape': (1, 15, 15, 12),
            'epochs': 12,
            'l2_reg': 0.001,
            'l_r': 1e-05,
            'nb_cnn_layers': 3,
            'nb_dense_layers': 3,
        }
    """
    seq_model = Sequential()

    w_init = initializers.TruncatedNormal(mean=0, stddev=0.01)
    k_regulizer = regularizers.l2(params['l2_reg'])

    for i in range(1, params['nb_cnn_layers'] + 1):

        seq_model.add(Conv3D(
            params[f'__cnn_neurons_{i}'],
            (1, params[f'__cnn_ksize_{i}'], params[f'__cnn_ksize_{i}']),
            padding='VALID',
            activation=params['activation_cnn'],
            input_shape=(params['dshape'][0], params['dshape'][1],
                         params['dshape'][2], params['dshape'][3]),
            kernel_initializer=w_init,
            kernel_regularizer=k_regulizer,
            strides=1
        ))
        if params['batch_norm']:
            seq_model.add(BatchNormalization())
        dropout_val = params[f'__cnn_dropout_{i}']
        if dropout_val > 0.01:
            seq_model.add(Dropout(dropout_val))

    seq_model.add(Flatten())

    for i in range(1, params['nb_dense_layers'] + 1):
        seq_model.add(Dense(
            params[f'__dense_neurons_{i}'],
            activation=params['activation_dense'],
            kernel_initializer=w_init,
            kernel_regularizer=k_regulizer
        ))

        if params['batch_norm']:
            seq_model.add(BatchNormalization())

        dropout_val = params[f'__dense_dropout_{i}']
        if dropout_val > 0.01:
            seq_model.add(Dropout(dropout_val))

    seq_model.add(Dense(
        nb_classes,
        activation='softmax',
        kernel_initializer=w_init,
        kernel_regularizer=k_regulizer
    ))

    seq_model.compile(loss='categorical_crossentropy',
                      optimizer=adam(lr=params['l_r']),
                      metrics=['accuracy'])

    if summary:
        seq_model.summary()

    return seq_model

def train_model(generator, num_classes, params, class_weight, steps_per_epoch,
                valid_data, valid_labels, verbose=0):
    """Train model function."""

    keras_model = conv3d_model(num_classes, params, summary=False)
    # early_stopper = EarlyStopping(monitor='val_accuracy', min_delta=0.1,
    #                               patience=5, verbose=0, mode='auto')

    history = keras_model.fit_generator(
        generator,
        steps_per_epoch=steps_per_epoch,
        epochs=params['epochs'],
        verbose=verbose,
        validation_data=(valid_data, valid_labels),
        class_weight=class_weight,
        shuffle=True,
        workers=8,
        max_queue_size=int(steps_per_epoch * 2.2) # How many batches in the queue
        # callbacks=[early_stopper]
    )
    return history, keras_model
