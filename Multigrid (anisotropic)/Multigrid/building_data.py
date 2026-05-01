''' This file imports the image and builds the list of images:
- "import_and_reshape_image" imports an image as an array and rehsapes it to (2**biggest_size,2**biggest_size)
- "halve" halves an image "times" times
- "add_gaussian_noise"
- "build_list_of_images" builds an array of length sublevels+1 of type 'dtype' whose first element is image and the next following ones are the image progressively "halved"
'''

import numpy as np
from PIL import Img

def import_and_reshape_image(name, biggest_size, normalize=True, maxval=255, stretch=False):
    image = Img.open(name) # Image provides its own class/object. "image" is now 600 x 601
    image = image.convert('L') # convert to greyscale ('F') indicates 32-bit float pixels
    if 2**biggest_size > np.shape(np.asarray(image))[0]:
        raise Exception('The biggest grid should be smaller or equal than the image')
    image = np.asarray(image.resize((2**biggest_size,2**biggest_size)))
    if normalize is True:
        image = image/maxval
    if stretch is True:
        a, b = np.min(image), np.max(image)
        image = 1/(b-a) * (image - a)
    return image


def halve_gpt(M):
    """
    Halve a 2N x 2N NumPy array by averaging each non-overlapping 2x2 block.

    Parameters:
    M (np.ndarray): Input 2N x 2N matrix.

    Returns:
    np.ndarray: Halved N x N matrix where each element is the average of a 2x2 block from M.
    """
    # Ensure the input matrix has even dimensions
    if M.shape[0] % 2 != 0 or M.shape[1] % 2 != 0:
        raise ValueError("Both dimensions of M must be even.")

    # Reshape and compute the mean
    M_halved = M.reshape(M.shape[0]//2, 2, M.shape[1]//2, 2).mean(axis=(1, 3))
    return M_halved


def halve(im, times=1):
    if times==0:
        return im
    if times==1:
        m = im.shape[0]
        if m%2==1:
            raise Exception('You are trying to halve an image with odd side')
        m = int(m/2)
        im_halved = np.zeros((m,m))
        for i in range(m):
            for j in range(m):
                im_halved[i,j] = np.mean([im[2*i,2*j], im[2*i+1,2*j], im[2*i,2*j+1], im[2*i+1,2*j+1]])
        return im_halved
    else:
        return halve(halve(im,1), times-1)

def average(im, times=1): # averages a matrix; the matrix has side which is a power of 2
    if times == 0:
        return im
    x = np.zeros(np.shape(im))
    h = halve(im,times)
    for i in range(2**times):
        for j in range(2**times):
            x[i::2**times,j::2**times] = h
    return x
            

# def add_gaussian_noise(image, var=100): # other kinds of noise available at https://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv
#     row,col = image.shape
#     mean = 0
#     sigma = np.sqrt(var)
#     noise = np.random.normal(mean,sigma,(row,col))
#     noisy = image + noise
#     return noisy

def add_gaussian_noise(image, sig=0.1):
    row, col = image.shape
    return image + sig * np.random.normal(size=(row,col))

def build_list_of_images(image, sublevels):
    if sublevels > np.log2(image.shape[0]):
        raise Exception('You are asking to build a list with too many sublevels')
    list1 = [image]
    for i in range(sublevels):
        list1 = list1 + [halve(list1[-1])]
    return np.array(list1, dtype=object)