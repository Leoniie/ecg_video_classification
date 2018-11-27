import os

import numpy as np
from keras.utils import to_categorical

from helpers.io import inputter_csv_file, inputter_videos_from_folder, outputter
from helpers.preprocessing import preprocessing, max_time
from models.Sequential import evaluate_sequential

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

dir_path = os.path.dirname(os.path.realpath(__file__))

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

# Model
#  y_train = to_categorical(y_train)

max_time_steps = np.max((max_time(x_train), max_time(x_test)))

y = evaluate_sequential(preprocessing(x_train, max_time_steps, normalizing=True,
                                      scaling=True, resolution_type='resize', resolution=0.5),
                        to_categorical(y_train),
                        preprocessing(x_test, max_time_steps, normalizing=True,
                                      scaling=True, resolution_type='resize', resolution=0.5))

outputter(y)

# plt.figure()
# plt.imshow(a)
# plt.savefig("small2.png")

# x_train_full = preprocessing(x_train)
# x_train_scaled = preprocessing_scaled(x_train)
