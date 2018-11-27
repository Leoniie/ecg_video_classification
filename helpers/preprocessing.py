import numpy as np
from skimage.transform import rescale, resize, downscale_local_mean


def scaling(df, resolution_type = 'resize', resolution = 0.5):
    """
    :param df: 5d-array [sample, steps, height, width, channels]
    :param resolution_type: resize, rescale, downsize
    :param resolution: float (0,1)
    """

    # change resolution
    if resolution_type == 'resize':
        df = resize(df, (df.shape[0], df.shape[1], int(df.shape[2] * resolution),
                         int(df.shape[3] * resolution), df.shape[1]),
                    mode='constant')
    elif resolution_type == 'rescale':
        df = rescale(df, (df.shape[0], df.shape[1], int(df.shape[2] * resolution),
                         int(df.shape[3] * resolution), df.shape[1]),
                     mode='constant')
    elif resolution_type == 'downscale':
        df = downscale_local_mean(df, (df.shape[0], df.shape[1], int(df.shape[2] * resolution),
                         int(df.shape[3] * resolution), df.shape[1]),
                                mode='constant')
    else:
        print('Wrong resolution_type detected')

    return df

# leonie
def cropping():
    pass
    # TODO: reduce the image to moving pixels

def normalize():
    pass
    # done in preprocessing anyway

    # possibility to reduce values from 0 to 255

def preprocessing(x,max_time):
    #input: list of 3d arrays with shape (timesteps, height, width)
    #also includes normalization

    x_array = np.zeros((x.shape[0], max_time, x[0].shape[1], x[0].shape[2]))

    for i in np.arange(x.shape[0]):
        v = x[i]
        d = np.max(abs(v))
        np.swapaxes(v, 1, 2)
        np.swapaxes(v, 0, 1)
        v = v / d
        x_array[i, :v.shape[0], :v.shape[1], :v.shape[2]] = v

    x_array = np.resize(x_array, (x_array.shape[0], x_array.shape[1], x_array.shape[2], x_array.shape[3], 1))

    return x_array

def max_time(x):
    
    max_time = 0
    for i in np.arange(x.shape[0]):
        max_time = np.max((max_time, (x[i]).shape[0]))

    return max_time



def preprocessing_scaled(x,max_time):
    #input: list of 3d arrays with shape (timesteps, height, width)
    #also includes normalization


    x_array = np.zeros((x.shape[0], max_time, 50, 50))


def preprocessing_scaled(x, max_time):
    # input: list of 3d arrays with shape (timesteps, height, width)
    # also includes normalization

    x_array = np.zeros((x.shape[0], max_time, 50, 50))

    for i in np.arange(x.shape[0]):
        v = x[i]
        v_scaled = np.zeros((v.shape[0], 50, 50))
        for j in np.arange(v_scaled.shape[0]):
            a = np.zeros((50, 50))
            image = v[j, :, :]
            for l in np.arange(50):
                for k in np.arange(50):
                    a[l, k] = np.sum(image[2 * l:2 * l + 2, 2 * k:2 * k + 2])
            v_scaled[j, :, :] = a
        d = np.max(abs(v_scaled))
        v_scaled = v_scaled/d
        np.swapaxes(v,1,2)
        np.swapaxes(v,0,1)
        x_array[i,:v_scaled.shape[0], :v_scaled.shape[1], :v_scaled.shape[2]]= v_scaled


    x_array = np.resize(x_array, (x_array.shape[0], x_array.shape[1], x_array.shape[2], x_array.shape[3], 1))

    return x_array




    #add the relevant arguments to the function
    # TODO: add binary decision variables
    # TODO: add the substeps to the function
    # TODO: return the preprocessed data set
    # TODO: add Exceptions


# be aware, no hard coding to be adaptable for train and test data
