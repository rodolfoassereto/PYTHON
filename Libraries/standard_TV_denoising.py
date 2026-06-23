import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import scipy.signal
from differential_operators import nabla, div_x

'''This function is already available in libraries'''
'''
def image_to_greyscale_array(image_path="C:/Users/rodol/My Drive/PYHTON/Images Licence free/abstract_pattern.jpg"):
    """
    Loads an image, converts it to greyscale, and normalizes pixel values between 0 and 1.

    :param image_path: Path to the image file.
    :return: NumPy array of the greyscale image with values in range [0, 1].
    """
    # Open the image and convert to greyscale
    image = Image.open(image_path).convert("L")
    
    # Convert to NumPy array
    grey_array = np.array(image, dtype=np.float32)
    
    # Normalize pixel values to range [0, 1]
    grey_array /= 255.0
    
    return grey_array
'''

build_kernel = lambda size: np.ones((size, size), dtype=np.float32) / (size ** 2)

blur = lambda img, kernel: scipy.signal.convolve2d(img, kernel, mode='same', boundary='symm')

# pattern = image_to_greyscale_array()
# kernel = build_kernel(15)
# pattern_blurred = blur(pattern, kernel)
# pattern_noisy = pattern + np.random.normal(0, 0.2, pattern.shape)


def L2_TV(b, alpha, maxit=5000):
    
    from algorithms_general import CP
    from prox_and_proj import proj_ball_L_2_infty
    
    prox_f = lambda X, tau: { 'x': (X['x'] + tau*b) / (1+ tau) }
    prox_gstar = lambda Y, sig: { 'y': proj_ball_L_2_infty(Y['y'], lam=alpha, ax=0) }
    
    L = lambda X: { 'y': nabla(X['x']) }
    Lstar = lambda Y: {'x': -div_x(Y['y']) }
    
    X = {'x': b.copy()}
    Y = {'y': nabla(b)}
    
    sig, tau = 2 * [1 / np.sqrt(8)]
    
    X_out, Y_out = CP(X, Y, tau, sig, prox_f, prox_gstar, L, Lstar, maxit, printprogress=True)
    
    return X_out['x']

def L1_TVL1(b , alpha, iterations=5000):

    from algorithms_general import CP
    from prox_and_proj import proj_L_infty_ball, prox_norm1
    
    prox_f = lambda X, tau: { 'x': b + prox_norm1( X['x'] - b , tau ) }
    prox_gstar = lambda Y, sig: { 'y': proj_L_infty_ball(Y['y'], lam=alpha) }
    
    L = lambda X: { 'y': nabla(X['x']) }
    Lstar = lambda Y: {'x': -div_x(Y['y']) }
    
    X = {'x': b.copy()}
    Y = {'y': nabla(b)}
    
    sig, tau = 2 * [1 / np.sqrt(8) ]
    
    X_out, Y_out = CP(X, Y, tau, sig, prox_f, prox_gstar, L, Lstar, iterations, printprogress=True)
    
    return X_out['x']

def L2(x):
    return 0.5* np.sum(x**2)
def L1(x):
    return np.sum( np.abs(x) )
def TVL2(x):
    return np.sum( np.linalg.norm( nabla(x), axis=0) )
def TVL1(x):
    return np.sum( np.abs(nabla(x)) )


if __name__ == 'main':
    img = np.zeros((7,7))
    img[:3,:3] = 1
    img[:3,3:] = 0.666
    img[3:,:3] = 0.333
    plt.imshow(img, cmap='gray', vmin=0, vmax=1)
    plt.show()

    plt.close('all')
    img_reg = L1_TVL1(img, 10, iterations=40000)
    plt.imshow(img_reg, cmap='gray', vmin=0, vmax=1)
    plt.show()