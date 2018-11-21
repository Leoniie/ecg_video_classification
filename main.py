import os
import tensorflow as tf

from aml_example_files.tf_utils import save_tf_record, prob_positive_class_from_prediction, input_fn_from_dataset
from helpers.io import inputter_csv_file, inputter_videos_from_folder, outputter


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

dir_path = os.path.dirname(os.path.realpath(__file__))

# Input folders

train_folder = os.path.join(dir_path,"data\\train")
test_folder = os.path.join(dir_path," gdata\\test")

# Create tf records in super folder for train records outside git-repository

tf_record_dir = os.path.join(dir_path,'tf_records')
os.makedirs(tf_record_dir, exist_ok=True)

tf_record_train = os.path.join(tf_record_dir, 'train' + '.tfrecords')
tf_record_test = os.path.join(tf_record_dir, 'test' + '.tfrecords')

if not os.path.exists(tf_record_train):
	x_train = inputter_videos_from_folder(train_folder)
	y_train = inputter_csv_file(dir_path, 'data\\train_target.csv')
	save_tf_record(x_train,tf_record_train,y = y_train)

if not os.path.exists(tf_record_test):
	x_test = inputter_videos_from_folder(test_folder)
	save_tf_record(x_test,tf_record_test)

# Model


## Example from
feature_col_video = tf.feature_column.numeric_column('video',shape=[212,100,100],dtype=tf.uint8)
feature_col_frame = tf.feature_column.numeric_column('frame',shape = [100,100],dtype=tf.uint8)
batchsize_video = 1

estimator = tf.estimator.LinearClassifier(feature_columns = [feature_col_video],model_dir='tf_checkpoints')
estimator.train(input_fn=lambda: input_fn_from_dataset(tf_record_train, batch_size=batchsize_video),max_steps=1)
pred = estimator.predict(input_fn=lambda: input_fn_from_dataset(tf_record_test, batch_size=batchsize_video,num_epochs=1,shuffle = False))
y_sol = prob_positive_class_from_prediction(pred)

# Generate output

#outputter(y_sol)