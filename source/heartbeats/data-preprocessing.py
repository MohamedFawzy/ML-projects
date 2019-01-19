from __future__ import print_function
from scipy.io import wavfile
import os
from shutil import copyfile
import csv
import os
import matplotlib

matplotlib.use('Agg')  # No pictures displayed
import pylab
import librosa
import librosa.display

MAIN_PATH = 'data/train/'
WAV_MAIN_PATH = 'data/samples/train/'


def generateMfccFeatures(filepath):
    """

    :param filepath:
    :return:
    """
    y, sr = librosa.load(filepath)
    mfcc_features = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return mfcc_features


def savetoSpectrogram(mfcc_features, filename):
    """

    :param mfcc_features:
    :param filename:
    :return:
    """
    pylab.figure(figsize=(10, 4))
    pylab.axis('off')
    pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])  # Remove the white edge
    librosa.display.specshow(mfcc_features, cmap='Set1')
    pylab.savefig(filename, bbox_inches=None, pad_inches=0)
    pylab.close()


def makeDir():
    """
    create main path dir
    :return:
    """
    if not os.path.exists(MAIN_PATH):
        print("Creating directory ", MAIN_PATH, ".....")
        os.mkdir(MAIN_PATH)
        os.mkdir(MAIN_PATH + 'normal')
        os.mkdir(MAIN_PATH + 'abnormal')


def loadData():
    counter = 0
    for root, dirs, files in os.walk(WAV_MAIN_PATH):
        for file in files:
            if file.endswith('.wav'):
                class_name = os.path.basename(root)
                file_with_path = os.path.join(root, file)
                print("Generating MFCC features for file ", file_with_path)
                mfcc_features = generateMfccFeatures(file_with_path)
                new_file = MAIN_PATH + class_name + '/' + file + '.jpg'
                print("Saving MFCC features to spectrogram in new file ", new_file)
                savetoSpectrogram(mfcc_features, new_file)
                counter = counter + 1
                print("Iteration # ", counter)
                print("=========================================\n")

    print("Finishing for all files ", counter)


def main():
    makeDir()
    loadData()


if __name__ == '__main__':
    main()
