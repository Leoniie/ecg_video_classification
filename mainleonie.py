import os
import tensorflow as tf

from aml_example_files.tf_utils import save_tf_record, prob_positive_class_from_prediction, input_fn_from_dataset
from helpers.io import inputter_csv_file, inputter_videos_from_folder, outputter




# Model
dir_path = os.path.dirname(os.path.realpath(__file__))

