import numpy as np
from skimage.transform import rescale, resize, downscale_local_mean


def scale(df, resolution_type='resize', resolution=0.5):
    """
    :param df: 5d-array [sample, steps, height, width, channels]
    :param resolution_type: resize, rescale, downsize
    :param resolution: float (0,1)
    """

    # TODO: Changing resolution does not work for video YET. FUUCK

    # change resolution
    if resolution_type == 'resize':
        df = resize(df, (df.shape[0], df.shape[1], int(df.shape[2] * resolution),
                         int(df.shape[3] * resolution), df.shape[4]),
                    mode='constant')
    elif resolution_type == 'rescale':
        df = rescale(df, (df.shape[0], df.shape[1], int(df.shape[2] * resolution),
                          int(df.shape[3] * resolution), df.shape[4]),
                     mode='constant')
    elif resolution_type == 'downscale':
        df = downscale_local_mean(df, (df.shape[0], df.shape[1], int(df.shape[2] * resolution),
                                       int(df.shape[3] * resolution), df.shape[4]))
    else:
        print('Wrong resolution_type detected')

    return df


# leonie
def cropping():
    pass
    # TODO: reduce the image to moving pixels


def normalize(df):
    df_max_frame = np.max(abs(df), 0)
    df = df / df_max_frame ###Hier hat der ein problem: invalid value encountered
    return df


def list_to_array(x_data, maxtime):
    x_array = np.zeros((x_data.shape[0], maxtime, x_data[0].shape[1], x_data[0].shape[2]))

    for i in np.arange(x_data.shape[0]):
        v = x_data[i]
        x_array[i, :v.shape[0], :v.shape[1], :v.shape[2]] = v

    # swap time axis from 3rd to 2nd dimension
    np.swapaxes(x_array, 2, 3)
    np.swapaxes(x_array, 2, 1)

    x_array = np.resize(x_array, (x_array.shape[0], x_array.shape[1], x_array.shape[2], x_array.shape[3], 1))

    return x_array


def max_time(x):
    maxtime = 0
    for i in np.arange(x.shape[0]):
        maxtime = np.max((maxtime, (x[i]).shape[0]))

    return maxtime


def preprocessing(x_data, max_time, normalizing=True, scaling=True, resolution_type='resize', resolution=0.5):
    df = list_to_array(x_data, max_time)
    if normalizing:
        df = normalize(df)
    if scaling:
        df = scale(df, resolution_type, resolution)

    return df

    # TODO: add Exceptions
