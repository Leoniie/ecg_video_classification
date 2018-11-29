import os

import numpy as np
from keras.utils import to_categorical

from helpers.io import inputter_csv_file, inputter_videos_from_folder, outputter
from helpers.preprocessing import preprocessing, max_time
from models.Sequential_Conv3D import evaluate_sequential

# from helpers.output import output_generator

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


x_train = inputter_videos_from_folder(train_folder)
y_train = inputter_csv_file(dir_path, 'data/train_target.csv')


x_test = inputter_videos_from_folder(test_folder)


max_time_steps = np.max((max_time(x_test),max_time(x_train)))

# Model
#y_train = to_categorical(y_train)

max_time_steps = np.max((max_time(x_train),max_time(x_test)))

x_train = preprocessing(x_train, max_time_steps, normalizing=True,
                                      scaling=False, resolution=0.5, cut_time=True, length=50)
x_test = preprocessing(x_test, max_time_steps, normalizing=True,
                                      scaling=False, resolution=0.5, cut_time=True, length=50)
y_train = to_categorical(y_train)

# !!! scaling funktioniert noch nicht !!!
y = evaluate_sequential(x_train,
                        y_train,
                        x_test)

outputter(y)

# plt.figure()
# plt.imshow(a)
# plt.savefig("small2.png")

#x_train_full = preprocessing(x_train)
#x_train_scaled = preprocessing_scaled(x_train)