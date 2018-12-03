

import numpy as np
import os

from helpers.io import inputter_csv_file, inputter_videos_from_folder, outputter
from helpers.preprocessing import preprocessing, max_time, cropping, gaussian_filtering, edge_filter, min_time

from models.Sequential_Conv3D import evaluate_sequential
from models.autoencoder import evaluate_auto

# from helpers.output import output_generator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

dir_path = os.path.dirname(os.path.realpath(__file__))

PREPROCESSING = True
# Preprocessing parameter
RESOLUTION = 1.0



if PREPROCESSING:
    # Input folders

    train_folder = os.path.join(dir_path, "data/train/")
    test_folder = os.path.join(dir_path, "data/test/")

    # Create tf records in super folder for train records outside git-repository

    tf_record_dir = os.path.join(dir_path, 'tf_records')
    os.makedirs(tf_record_dir, exist_ok=True)

    tf_record_train = os.path.join(tf_record_dir, 'train' + '.tfrecords')
    tf_record_test = os.path.join(tf_record_dir, 'test' + '.tfrecords')

    x_train = inputter_videos_from_folder(train_folder)
    y_train = inputter_csv_file(dir_path, 'data/train_target.csv')

    x_test = inputter_videos_from_folder(test_folder)

    max_time_steps = np.max((max_time(x_train), max_time(x_test)))

    min_time_steps = np.min((min_time(x_train), min_time(x_test)))




    x_train = preprocessing(x_train, max_time_steps, normalizing=False,
                            scaling=True, resolution=RESOLUTION, cut_time=True, length = 20, crop=0, filter=False)
    x_test = preprocessing(x_test, max_time_steps, normalizing=False,
                           scaling=True, resolution=RESOLUTION, cut_time=True, length= 20, crop=0, filter=False)

else:

    y_train = np.load('data/numpy/y_train.npy')
    print("Loaded: y_train with shape {}".format(y_train.shape))
    x_train = np.load('data/numpy/x_train.npy')
    print("Loaded: x_train with shape {}".format(x_train.shape))
    x_test = np.load('data/numpy/x_test.npy')
    print("Loaded: x_test with shape {}".format(x_test.shape))

x_train_en = evaluate_auto(x_train)
x_test_en = evaluate_auto(x_test)


y = evaluate_sequential(x_train_en,
                        y_train,
                        x_test_en)

outputter(y)