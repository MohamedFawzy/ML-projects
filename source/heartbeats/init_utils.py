import os
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras import optimizers
from keras import metrics
from keras.callbacks import EarlyStopping, History, ModelCheckpoint
from keras.layers.core import Flatten, Dense, Dropout, Reshape, Lambda
from keras.layers.normalization import BatchNormalization
import matplotlib
from IPython.display import Image
from IPython.display import display

import matplotlib.pyplot as plt
import keras.backend as K

IMG_WIDTH, IMG_HEIGHT = 150, 150

train_data_path = 'data/t'
validation_data_path = 'data/v'

normal_validation_dir = 'data/v/normal/'
normal_training_dir = 'data/t/normal/'

abnormal_validation_dir = 'data/v/abnormal/'
abnormal_training_dir = 'data/t/abnormal/'


def get_sample_spectrogram():
    """
    :return: list of spectrogram png files to display them
    """
    Images = [
        'data/t/normal/e00008.png',
        'data/v/normal/e00002.png',
        'data/t/abnormal/e00304.png',
        'data/v/abnormal/e00435.png'
    ]

    for imageName in Images:
        display(Image(filename=imageName))


def get_images_number_per_class():
    """
    count number of files in each directory as normal or abnormal in training and validation set
    :return: number of training validation images as normal/abnormal
    """
    normal_train = len(next(os.walk(normal_training_dir))[2])
    normal_validation = len(next(os.walk(normal_validation_dir))[2])
    abnormal_train = len(next(os.walk(abnormal_training_dir))[2])
    abnormal_validation = len(next(os.walk(abnormal_validation_dir))[2])

    ## We subtract 1 since every directory has a hidden file .DS_Store
    print ('Number of samples in training set (normal): {}'.format((normal_train) - 1))
    print ('Number of samples in validation set (normal): {}'.format((normal_validation) - 1))
    print ('Number of samples in training set (abnormal): {}'.format((abnormal_train) - 1))
    print ('Number of samples in validation set (abnormal): {}'.format((abnormal_validation) - 1))

    return normal_train, normal_validation, abnormal_train, abnormal_validation


def visualize_samples(normal_train, normal_validation, abnormal_train, abnormal_validation):
    """

    :return: visualize graph for training / validation set with it's classes
    """
    n_groups = 2

    train_samples = (normal_train, normal_validation)
    validation_samples = (abnormal_train, abnormal_validation)

    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.4
    opacity = 0.7

    fig1 = plt.bar(index, train_samples, bar_width,
                   alpha=opacity,
                   color='r',
                   label='normal')

    fig2 = plt.bar(index + bar_width, validation_samples, bar_width,
                   alpha=opacity,
                   color='b',
                   label='abnormal')

    plt.xlabel('Classes')
    plt.ylabel('Number of Samples')
    plt.title('Samples and Classes')
    plt.xticks(index + bar_width, ('Train', 'Validation'))
    plt.legend()

    plt.tight_layout()
    plt.show()


def data_augmentation():
    """
        generate images from current images to feed the network with larger dataset to help it for better training data
        :return: generated images from data
    """
    data_generation = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.4,
        zoom_range=0.3,
        horizontal_flip=True
    )

    train_data_generator = data_generation.flow_from_directory(
        train_data_path,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=64,
        class_mode='binary'
    )

    validation_data_generator = data_generation.flow_from_directory(
        validation_data_path,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=64,
        class_mode='binary'
    )

    return train_data_generator, validation_data_generator


def precision(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return: precision number
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def fbeta_score(y_true, y_pred, beta=1):
    """

    :param y_true:
    :param y_pred:
    :param beta:
    :return:
    """
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score


def results(history):
    """

    :param history:
    :return: figure for accuracy and loss
    """
    # Accuracy
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['acc']);
    plt.plot(history.history['val_acc']);
    plt.title('model accuracy');
    plt.ylabel('accuracy');
    plt.xlabel('epoch');
    plt.legend(['train', 'valid'], loc='upper left');

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss']);
    plt.plot(history.history['val_loss']);
    plt.title('model loss');
    plt.ylabel('loss');
    plt.xlabel('epoch');
    plt.legend(['train', 'valid'], loc='upper left');
    plt.show()


model = Sequential()


def small_cnn(train_data, validation_data, n_epoch=5, n_train_samples=1000, n_validation_samples=255):
    # # layer 1
    model.add(Convolution2D(32, kernel_size=(3, 3), input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # layer 2
    model.add(Convolution2D(64, kernel_size=(3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # layer 3
    model.add(Convolution2D(128, kernel_size=(3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # layer 4
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', precision, recall, fbeta_score])

    return model.fit_generator(train_data, samples_per_epoch=n_train_samples, nb_epoch=n_epoch,
                               validation_data=validation_data, nb_val_samples=n_validation_samples)


def evaluate(validation_data, validation_samples):
    return model.evaluate_generator(validation_data, validation_samples)


mod = Sequential()
def deeper_cnn(train_data, validation_data, n_epoch=5, n_train_samples=1000, n_validation_samples=255):
    """

    :param train_data:
    :param validation_data:
    :param n_epoch:
    :param n_train_samples:
    :param n_validation_samples:
    :return: trained network with classification
    """

    # first layer
    mod.add(Convolution2D(32, kernel_size=(3, 3), use_bias=True))
    mod.add(ZeroPadding2D((1, 1), input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)))
    mod.add(BatchNormalization())
    mod.add(Activation("relu"))
    mod.add(Dropout(0.5))
    mod.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # 2nd layer
    mod.add(Convolution2D(64, kernel_size=(3, 3), use_bias=True))
    mod.add(ZeroPadding2D((1, 1), input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)))
    mod.add(BatchNormalization())
    mod.add(Activation('relu'))
    mod.add(Dropout(0.5))
    mod.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    mod.add(Convolution2D(128, kernel_size=(3, 3), use_bias=True))
    mod.add(ZeroPadding2D((1, 1), input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)))
    mod.add(BatchNormalization())
    mod.add(Activation("relu"))
    mod.add(Dropout(0.5))
    mod.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))


    mod.add(Convolution2D(256, kernel_size=(3, 3), use_bias=True))
    mod.add(ZeroPadding2D((1, 1), input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)))
    mod.add(BatchNormalization())
    mod.add(Activation("relu"))
    mod.add(Dropout(0.5))
    mod.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))


    mod.add(Convolution2D(512, kernel_size=(3, 3), use_bias=True))
    mod.add(ZeroPadding2D((1, 1), input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)))
    mod.add(BatchNormalization())
    mod.add(Activation("relu"))
    mod.add(Dropout(0.5))
    mod.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    mod.add(Convolution2D(512, kernel_size=(3, 3), use_bias=True))
    mod.add(ZeroPadding2D((1, 1), input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)))
    mod.add(BatchNormalization())
    mod.add(Activation("relu"))
    mod.add(Dropout(0.5))
    mod.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))


    mod.add(Flatten())
    mod.add(Dense(1024))
    mod.add(BatchNormalization())
    mod.add(Activation('relu'))
    mod.add(Dropout(0.5))
    mod.add(Dense(1))
    mod.add(BatchNormalization())
    mod.add(Activation('sigmoid'))



    mod.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', precision, recall, fbeta_score])

    return mod.fit_generator(train_data, samples_per_epoch=n_train_samples, nb_epoch=n_epoch,
                               validation_data=validation_data, nb_val_samples=n_validation_samples)