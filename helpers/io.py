import numpy as np
import skvideo.io
import os
import pandas as pd
from datetime import datetime

dir_path = os.path.dirname(os.path.realpath(__file__))


def inputter_videos_from_folder(data_folder):
    '''
    get a list of video x wehre each video is a numpy array in the format [n_frames,width,height]
    with uint8 elements.
    argument: relative path to the data_folder from the source folder.
    '''
    data_folder = os.path.join(dir_path, data_folder)
    x = []
    file_names = []

    if os.path.isdir(data_folder):
        for dirpath, dirnames, filenames in os.walk(data_folder):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                statinfo = os.stat(file_path)
                if statinfo.st_size != 0:
                    video = skvideo.io.vread(file_path, outputdict={"-pix_fmt": "gray"})[:, :, :, 0]
                    x.append(video)
                    file_names.append(int(filename.split(".")[0]))

    indices = sorted(range(len(file_names)), key=file_names.__getitem__)
    x = np.take(x, indices)
    return x


def inputter_csv_file(dir_path, csv_file):
    '''
    get a numpy array y of labels. the order follows the id of video.
    argument: relative path to the csv_file from the source folder.
    '''
    csv_file = os.path.join(dir_path, csv_file)
    with open(csv_file, 'r') as csvfile:
        label_reader = pd.read_csv(csvfile)
        y = label_reader['y']

    y = np.array(y)
    return y


def outputter(array):
    y = pd.DataFrame(array, dtype=np.dtype('U25'))
    ids = list(range(0, y.shape[0]))
    ids = pd.DataFrame(ids)
    output = pd.concat([ids, y], axis=1)
    output.columns = ["id", "y"]
    _dir = os.path.dirname(__file__)
    filename = os.path.join(_dir, '/output/solution.csv')
    now = datetime.today()
    print(now)
    s = "_"
    seq = (str(now), "solution.csv")
    print(seq)
    file_name = s.join(seq)  # type: str
    s = "\\"
    sequence = ("output", file_name)
    path = s.join(sequence)
    print(path)
    output.to_csv(path_or_buf='/output/solution.csv', sep=',', na_rep='', float_format='U25',
                  header=True, index=False,
                  mode='w', encoding=None, compression=None,
                  quoting=None, quotechar='"', line_terminator='\n', chunksize=None, tupleize_cols=None,
                  date_format=None, doublequote=True, escapechar=None, decimal='.')
