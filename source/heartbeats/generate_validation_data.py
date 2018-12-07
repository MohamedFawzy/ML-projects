from shutil import copyfile

path = 'data/validation/REFERENCE.csv'
training_normal_heart_sounds = 'data/t/normal/'
training_abnormal_heart_sounds = 'data/t/abnormal/'
validation_normal_heart_sounds = 'data/v/normal/'
validation_abnormal_heart_sounds = 'data/v/abnormal/'


def get_validtion_spectogram():
    with open(path, 'r') as fileName:
        lines = fileName.readlines()
        for line in lines:
            x = line.rstrip("\n").split(",")
            file_name = x[0]
            class_label = int(x[1])
            print(class_label)
            # file with -1 means that's normal hearbeat sounds so we take the spectogram from /t/normal otherwise it's abnormal heartsounds
            if (class_label == 1):
                print(file_name, class_label, 'if')
                copyfile(training_abnormal_heart_sounds + file_name + '.png',
                         validation_abnormal_heart_sounds + file_name + '.png')

            else:

                print(file_name, class_label, 'else')
                copyfile(training_normal_heart_sounds + file_name + '.png',
                         validation_normal_heart_sounds + file_name + '.png')


if __name__ == '__main__':
    get_validtion_spectogram()
