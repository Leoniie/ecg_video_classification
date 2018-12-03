

import numpy as np
import os
from models.conv2Dclassifier import to2D
from helpers.io import inputter_csv_file, inputter_videos_from_folder, outputter, outputter2, outputter3
from helpers.preprocessing import preprocessing, max_time, cropping, gaussian_filtering, edge_filter, min_time

from models.Sequential_Conv3D import evaluate_sequential
from models.autoencoder import evaluate_auto

# from helpers.output import output_generator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

dir_path = os.path.dirname(os.path.realpath(__file__))

PREPROCESSING = False
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
                            scaling=True, resolution=RESOLUTION, cut_time=True, length = 8, crop=0, filter=False, binary=True)
    x_test = preprocessing(x_test, max_time_steps, normalizing=False,
                           scaling=True, resolution=RESOLUTION, cut_time=True, length= 8, crop=0, filter=False, binary=True)

else:

    y_train = np.load('data/numpy/y_train.npy')
    print("Loaded: y_train with shape {}".format(y_train.shape))
    x_train = np.load('data/numpy/x_train.npy')
    print("Loaded: x_train with shape {}".format(x_train.shape))
    x_test = np.load('data/numpy/x_test.npy')
    print("Loaded: x_test with shape {}".format(x_test.shape))



a_train = np.zeros((158 * x_train.shape[1], x_train.shape[2], x_train.shape[3],1))
b = np.zeros((158 * x_train.shape[1]))
for i in range(x_train.shape[0]):
    for j in range(x_train.shape[1]):
        index = x_train.shape[1] * i + j
        a_train[index, :, :,0] = x_train[i, j, :, :, 0]
        b[index] = y_train[i]


a_test = np.zeros((x_test.shape[0] * x_test.shape[1], x_test.shape[2], x_test.shape[3],1))

for i in range(x_test.shape[0]):
    for j in range(x_test.shape[1]):
        index = x_test.shape[1] * i + j
        a_test[index, :, :,0] = x_test[i, j, :, :, 0]

a_train = evaluate_auto(a_train)
a_test = evaluate_auto(a_test)
b_test = to2D(a_train,b,a_test)
q = np.zeros((69))

b_test = np.round(b_test)

n_images = x_test.shape[1]

for i in range(69):
    for j in range(n_images):
        q[i] += b_test[i*n_images+j]

q = q/n_images
q = np.round(q)

outputter(q)
