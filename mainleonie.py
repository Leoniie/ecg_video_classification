import os
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, LSTM



from aml_example_files.tf_utils import save_tf_record, prob_positive_class_from_prediction, input_fn_from_dataset, save_x
from helpers.io import inputter_csv_file, inputter_videos_from_folder_array, outputter
from helpers.preprocessing import preprocessing, preprocessing_scaled

import matplotlib.pyplot as plt
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

dir_path = os.path.dirname(os.path.realpath(__file__))

# Input folders



train_folder = os.path.join(dir_path,"data/train/")
test_folder = os.path.join(dir_path,"data/test/")


# Create tf records in super folder for train records outside git-repository

tf_record_dir = os.path.join(dir_path,'tf_records')
os.makedirs(tf_record_dir, exist_ok=True)

tf_record_train = os.path.join(tf_record_dir, 'train' + '.tfrecords')
tf_record_test = os.path.join(tf_record_dir, 'test' + '.tfrecords')



x_train = inputter_videos_from_folder_array(train_folder)
y_train = inputter_csv_file(dir_path, 'data/train_target.csv')


x_test = inputter_videos_from_folder_array(test_folder)



x_train_full = preprocessing(x_train)
x_train_scaled = preprocessing_scaled(x_train)


3
4
5
6
7
8
model = Sequential()


model = Sequential()
model.add(TimeDistributed(Conv2D(1, (2,2), activation='relu', padding='same', input_shape=(100,100,1))))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
model.add(Flatten())
model.add(TimeDistributed(Conv2D(...))
model.add(TimeDistributed(MaxPooling2D(...)))
model.add(TimeDistributed(Flatten()))
# define LSTM model
model.add(LSTM(...))
model.add(Dense(...))