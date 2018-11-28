import numpy as np
from scipy import ndimage


def scale(df, resolution=0.5):
    """
    :param df: 5d-array [sample, steps, height, width, channels]
    :param resolution: float (0,1)
    """

    df = ndimage.zoom(input=df, zoom=(1, 1, resolution, resolution, 1), order=1)
    print("Scaled")

    return df


def normalize(df):
    df_max_frame = np.max(abs(df), axis=0)
    df = df / df_max_frame.astype(float)
    print("Normalized")
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

def cut_time_steps(x,length):
    x = x[:,:length, :, :, :]
    print("Length Cut")
    return x



def preprocessing( x_data, max_time, normalizing=True, scaling=True, resolution=0.5, cut_time=True,length=100):
    df = list_to_array(x_data, max_time)
    if cut_time:
        df = cut_time_steps(df, length)
    if normalizing:
        df = normalize(df)
    if scaling:
        df = scale(df, resolution)


    return df
