import os
import tensorflow as tf

from aml_example_files.tf_utils import save_tf_record, prob_positive_class_from_prediction, input_fn_from_dataset, save_x
from helpers.io import inputter_csv_file, inputter_videos_from_folder_array, outputter
from helpers.preprocessing import preprocessing, preprocessing_scaled

import matplotlib.pyplot as plt
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

dir_path = os.path.dirname(os.path.realpath(__file__))

# Input folders

train_folder = os.path.join(dir_path, "data/train/")
test_folder = os.path.join(dir_path, " gdata/test/")

# Create tf records in super folder for train records outside git-repository

tf_record_dir = os.path.join(dir_path,'tf_records')
os.makedirs(tf_record_dir, exist_ok=True)

tf_record_train = os.path.join(tf_record_dir, 'train' + '.tfrecords')
tf_record_test = os.path.join(tf_record_dir, 'test' + '.tfrecords')



x_train = inputter_videos_from_folder_array(train_folder)
y_train = inputter_csv_file(dir_path, 'data/train_target.csv')


x_test = inputter_videos_from_folder_array(test_folder)




# print(x_train[2].shape)
#
# print (x_train[2][0,:,:])
# # Model
#
# a = np.zeros((50,50))
#
# image = x_train[2][0,:,:]
#
# for i in np.arange(50):
#     for j in np.arange(50):
#         a[i,j]= np.sum(image[2*i:2*i+2,2*j:2*j+2])/4





#
# plt.figure()
# plt.imshow(a)
# plt.savefig("small2.png")

x_train_full = preprocessing(x_train)
x_train_scaled = preprocessing_scaled(x_train)
