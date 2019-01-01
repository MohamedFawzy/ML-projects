from scipy.io import wavfile
import os
from shutil import copyfile
import csv
from keras.utils.training_utils import multi_gpu_model

def move_to_dir(wav_file, class_label):
    file_name = os.path.basename(wav_file)

    if class_label == 1:
        print('Copy file ====> ' , 'data/t1/abnormal/'+file_name)
        copyfile(wav_file, 'data/t1/abnormal/'+file_name)
    else:
        print('Copy file ====> ', 'data/t1/normal/'+file_name)
        copyfile(wav_file, 'data/t1/normal/'+file_name)


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
    class_type = {"normal": 0, "abnormal": 1}

    number_of_classes = len(class_type.keys())

    data_path = 'data/training'
    wav_file_names = []
    class_labels = []

    samples = 0

    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith('.wav'):
                base_file_name = file.rstrip(".wav")
                label_file_name = os.path.join(root, base_file_name + ".hea")
                class_label = parse_class_label(label_file_name)
                class_labels.append(class_type[class_label])
                wav_file_names.append(os.path.join(root, file))
                samples += 1


    for wav_file_name, class_x in zip(wav_file_names, class_labels):
        move_to_dir(wav_file_name, class_x)


    # move the validation data set to validation folders as normal, abnormal also
    with open('data/validation/REFERENCE.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            file_name = row[0]
            class_type = int(row[1])
            if class_type == -1:
                print('Copy file ====> ', 'data/v1/normal/'+file_name)
                copyfile('data/validation/'+file_name+'.wav', 'data/v1/normal/'+file_name+'.wav')
            else:
                print('Copy file ====> ', 'data/v1/abnormal/'+file_name)
                copyfile('data/validation/'+file_name+'.wav', 'data/v1/abnormal/'+file_name+'.wav')


