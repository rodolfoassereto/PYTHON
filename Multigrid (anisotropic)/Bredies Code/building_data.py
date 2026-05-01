''' This file imports the image and builds the list of images:
- "import_image" imports an image as an array and rehsapes it to (2**biggest_size,2**biggest_size)
- "halve" halves an image "times" times
- "add_gaussian_noise"
- "build_list_of_images" builds an array of length sublevels+1 of type 'dtype' whose first element is image and the next following ones are the image progressively "halved"
'''

import numpy as np
from PIL import Image as Img

def import_image(name, biggest_size, normalize=True, maxval=255, stretch=False):
    image = Img.open(name) # Image provides its own class/object. "image" is now 600 x 601
    image = image.convert('F') # convert to greyscale ('F) indicates 32-bit float pixels
    if 2**biggest_size > np.shape(np.asarray(image))[0]: raise Exception('The biggest grid should be smaller or equal than the image')
    image = np.asarray(image.resize((2**biggest_size,2**biggest_size)))
    if normalize == True:
        image = image/maxval
    if stretch == True:
        a, b = np.min(image), np.max(image)
        image = 1/(b-a) * (image - a)
    return image

def halve(im, times=1):
    if times==0: return im
    if times==1:
        m = im.shape[0]
        if m%2==1: raise Exception('You are trying to halve an image with odd side')
        m = int(m/2)
        im_halved = np.zeros((m,m))
        for i in range(m):
            for j in range(m):
                im_halved[i,j] = np.mean([im[2*i,2*j], im[2*i+1,2*j], im[2*i,2*j+1], im[2*i+1,2*j+1]])
        return im_halved
    else:
        return halve(halve(im,1), times-1)

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
    if sublevels > np.log2(image.shape[0]): raise Exception('You are asking to build a list with too many sublevels')
    list1 = [image]
    for i in range(sublevels):
        list1 = list1 + [halve(list1[-1])]
    return np.array(list1, dtype=object)