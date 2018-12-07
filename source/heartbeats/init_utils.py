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
        rescale=1./255,
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
