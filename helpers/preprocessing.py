# TODO: Create preprocessing pipeline with plugable substeps
# possible library to use 'skvideo.utils'

def scaling():
    pass
    # TODO: use scikit video or cv2 to rescale the image/video
#leonie
def cropping():
    pass
    # TODO: reduce the image to moving pixels

def normalize():
    pass
    # TODO: add normalizer
    # possibility to reduce values from 0 to 255

def preprocessing():
    pass

    # TODO: add the relevant arguments to the function
    # TODO: add binary decision variables
    # TODO: add the substeps to the function
    # TODO: return the preprocessed data set
    # TODO: add Exceptions
    # be aware, no hard coding to be adaptable for train and test data
