import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os


def convert_to_spectrogram(wav_file, class_label):
    rate, data = get_wave_info(wav_file)
    nfft = 256  # Length of the windowing segments
    fs = 256  # Sampling frequency
    pxx, freqs, bins, im = plt.specgram(data, nfft, fs)

    plt.axis('off')
    file_name = os.path.basename(wav_file).rstrip(".wav")
    if (class_label == 1):
        print('saving file =====> ', 'data/abnormal/' + str(file_name) + '.png')
        plt.savefig('t/data/abnormal/' + str(file_name) + '.png',
                    dpi=100,  # Dots per inch
                    frameon='false',
                    aspect='normal',
                    bbox_inches='tight',
                    pad_inches=0)  # Spectrogram saved as a .png
    else:
        print('saving file ====> ', 'data/normal/' + str(file_name) + '.png')
        plt.savefig('t/data/normal/' + str(file_name) + '.png',
                    dpi=100,  # Dots per inch
                    frameon='false',
                    aspect='normal',
                    bbox_inches='tight',
                    pad_inches=0)


def get_wave_info(wav_file):
    rate, data = wavfile.read(wav_file)
    return rate, data


def parse_class_label(label_file_name):
    with open(label_file_name, 'r') as fileName:
        header = fileName.readlines()

    comments = [line for line in header if line.startswith("#")]

    if not len(comments) == 1:
        raise InvalidHeaderFileException("Invalid label file %s" % label_file_name)

    class_label = str(comments[0]).lstrip("#").rstrip("\r").strip().lower()

    if not class_label in class_type.keys():
        raise InvalidHeaderFileException("Invalid class label %s" % class_label)

    return class_label


if __name__ == '__main__':
    # set class type as normal heartbeats or abnormal heartbeats
    class_type = {"normal": 0, "abnormal": 1}
    # set number of classes
    number_of_classes = len(class_type.keys())
    # training data path
    data_path = 'data/training/'
    # wav file names
    wav_file_names = []
    # class labels
    class_labels = []
    # number of samples
    samples = 0

    ## processing all the recordings in 'training' folder
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith('.wav'):
                base_file_name = file.rstrip(".wav")
                label_file_name = os.path.join(root, base_file_name + ".hea")
                class_label = parse_class_label(label_file_name)
                class_labels.append(class_type[class_label])
                wav_file_names.append(os.path.join(root, file))
                samples += 1

    ## After saving all file name to lists, convering to spectrograms
    for wav_file_name, class_x in zip(wav_file_names, class_labels):
        convert_to_spectrogram(wav_file_name, class_x)

"""
data/training/training-e/e00927.wav 0
data/training/training-e/e00933.wav 0
data/training/training-e/e01393.wav 0
data/training/training-e/e00304.wav 1
data/training/training-a/a0371.wav 1
data/training/training-a/a0403.wav 1
data/training/training-a/a0365.wav 1
data/training/training-a/a0198.wav 1
"""
