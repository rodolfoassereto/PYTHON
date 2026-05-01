import numpy as np
from itertools import product

def partialplus(x, i): # you can use np.diff to make it possibly faster
    x = np.moveaxis(x, i, 0)
    y = np.zeros(x.shape)
    y[0:-1] = x[1:] - x[:-1]
    return np.moveaxis(y, 0, i)

def nabla_r(x): # works on the first two components, therefore works as standard "nabla" as well
    return np.array([ partialplus(x, 0), partialplus(x, 1) ])

''' Here is also a definition of TV for 1 and 2-dimensional arrays'''

tv1dim = lambda x: np.linalg.norm( partialplus(x, 0) )
tv2 = lambda x: np.sum( np.linalg.norm( nabla_r(x), axis=0 ) )
tvaniso = lambda x: np.sum(np.abs(nabla_r(x)))

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

def average(im, times=1): # averages a matrix; the matrix has side which is a power of 2
    if times == 0: return im
    x = np.zeros(np.shape(im))
    h = halve(im,times)
    for i in range(2**times):
        for j in range(2**times):
            x[i::2**times,j::2**times] = h
    return x

search_mat = np.zeros((6,6))
mask = np.zeros((6,6), dtype=bool)
mask[2:4,1:5] = True
mask[1:5,2:4] = True


import time
values = [0, 0.2, 0.4, 0.6, 0.8, 1]
t0 = time.time()
for v in product(values, repeat=12):
    if time.time() - t0 > 60:
        print(v)
        t0 = time.time()
    search_mat[mask] = v
    if tv2(search_mat) < tv2(average(search_mat)):
        print(search_mat)
        break
    
