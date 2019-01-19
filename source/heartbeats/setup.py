from __future__ import print_function
import os
from os.path import isfile
import glob
import pandas as pd
import shutil
import sys
from subprocess import call
import subprocess
from urllib.request import urlretrieve

main_path = 'data'
samples_path = 'samples/'
train_path = 'samples/train/'

class_names = ('normal', 'abnormal')

set_names = ('training-a', 'training-b', 'training-c', 'training-d', 'training-e', 'training-f')  # no point getting their val set

DEFAULT_DIR_NAME = 'training-f'
DEFAULT_FILE_NAME = 'training.zip'
DEFAULT_URL = 'https://www.physionet.org/physiobank/database/challenge/2016/training.zip'


def downloadIfNotExist(dir_name=DEFAULT_DIR_NAME, filename=DEFAULT_FILE_NAME, url=DEFAULT_URL, tar=False):
    """
    download data if not exist
    :param dir_name:
    :param filename:
    :param url:
    :param tar:
    :return:
    """
    if not os.path.isdir(dir_name):
        print("Directory ", dir_name, " not present . checking for compressed archive ", filename)

        if not os.path.isfile(filename):
            print("Compressed archive ", filename, " not present. Downloading it.... ")
            urlretrieve(url, filename)

        print(" Uncompressing archive......", end="")

        if (tar):
            call(['tar', '-zxf', filename])
        else:
            call(['unzip', filename])

        print(" done .")

    return


def makeDirs():
    """

    :return:
    """
    if not os.path.exists(main_path):
        print("Creating directory ", main_path, ".....")
        os.mkdir(main_path)
        print("Changing directory to ", main_path, " .....")
        os.chdir(main_path)

    if not os.path.exists(samples_path):
        os.mkdir(samples_path)
        os.mkdir(train_path)
        os.mkdir(test_path)
        for class_name in class_names:
            os.mkdir(train_path + class_name)
            os.mkdir(test_path + class_name)

    return


def readFile(path):
    """

    :param path:
    :return: string of files
    """
    line_list = open(path).readlines()
    return line_list


def main():
    makeDirs()
    downloadIfNotExist()

    for set_idx, set_name in enumerate(set_names):
        destpath = train_path

        ref_filename = set_name + '/' + 'REFERENCE.csv'
        df = pd.read_csv(ref_filename, names=("file", "code"))
        df["code"] = df["code"].replace([-1, 1], ['normal', 'abnormal'])
        df["file"] = df["file"].replace([r"$"], [".wav"], regex=True)

        for index, row in df.iterrows():
            this_file = row["file"]
            this_class = row["code"]
            src = set_name + '/' + this_file
            dst = destpath + this_class + '/cl-' + this_class + "-" + this_file
            print("src, dst =  ", src, dst)
            shutil.copyfile(src, dst)

    print("\nFINISHED.")
    return


if __name__ == '__main__':
    main()
