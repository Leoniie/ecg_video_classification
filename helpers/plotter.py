from scipy import ndimage, misc
import matplotlib.pyplot as plt


def plot(df):
    fig = plt.figure()
    plt.gray()  # show the filtered result in grayscale
    ax1 = fig.add_subplot(121)  # left side
    ax2 = fig.add_subplot(122)  # right side
    ax1.imshow(df[10,10,:,:,0])
    ax2.imshow(df[10,50,:,:,0])
    plt.show()

def plot3D(df):
    pass
