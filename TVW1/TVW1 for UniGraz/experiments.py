import numpy as np

'''
data: masked fft of gaussian mixture
problem: L2-TVPR
'''
def experiment_1():
    
    from build_toymodels import build_mask, gaussians, gaussian_mixture, plot_u
    from models import L2_TVPR
    
    mymask = build_mask(gaussian_mixture.shape, ratio=0.5, scheme='random')
    gaussian_mixture_b = np.ma.masked_array(np.fft.fftn(gaussian_mixture), mask=mymask)
    plot_u(np.real(np.fft.ifftn(np.ma.filled(gaussian_mixture_b, fill_value=0))))
    
    ustar_L2 = L2_TVPR(gaussian_mixture_b, forward=True, iterations=5000)
    plot_u(np.real(ustar_L2))


import cv2
import matplotlib.pyplot as plt
from curve_experiment import *
from models import *

np.random.seed(42)

def video_to_greyscale_array(video_path, resolution):
    """
    Convert a video to a T x N x N grayscale tensor with values between 0 and 1.

    Args:
        video_path (str): Path to the input MP4 video.
        resolution (tuple): Desired resolution (N, N) for each frame.

    Returns:
        np.ndarray: Grayscale tensor of shape (T, N, N) with values in range [0, 1].
    """
    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Resize frame to the desired resolution
        resized_frame = cv2.resize(gray_frame, resolution)
        # Normalize pixel values to range [0, 1]
        normalized_frame = resized_frame / 255.0
        # Append to the list of frames
        frames.append(normalized_frame)

    cap.release()

    # Convert list of frames to a NumPy array (T x N x N)
    tensor = np.array(frames)

    return tensor

def add_bugs_noise(tensor, maxradius, meannumber=2, meantime=2, return_masked=False):
    from curve_experiment import create_ball_matrix
    np.random.seed(42)
    tensor_out = tensor.copy()
    p_time = 1/meantime
    p_num = 1/meannumber
    T = tensor.shape[0]
    shapeimg = tensor.shape[1:]
    bugs_mask = np.zeros(tensor.shape, dtype=bool)
    for t in range(T):
        number_of_bugs = np.random.geometric(p_num)-1
        flat_indices = np.random.choice(np.prod(shapeimg), size=number_of_bugs, replace=False)
        # Convert flat indices to 2D coordinates
        row_indices, col_indices = np.unravel_index(flat_indices, shapeimg)
        # Combine row and column indices into list of tuples
        coordinates = list(zip(row_indices, col_indices))
        for ind in coordinates:
            radius = np.random.uniform(high=maxradius)
            duration = np.random.geometric(p_time)
            bugs_mask[t:t+duration] = bugs_mask[t:t+duration] | create_ball_matrix(shapeimg, ind, radius, output=bool)
    if return_masked: return np.ma.masked_array(tensor_out, bugs_mask)
    tensor_out[bugs_mask] = 0
    return tensor_out

import numpy as np

def add_salt_and_pepper_noise(image, amount=0.05, salt_vs_pepper=0.5):
    """
    Add salt and pepper noise to a grayscale image.

    Parameters:
    image (numpy.ndarray): Grayscale image data (2D array).
    amount (float): Proportion of image pixels to replace with noise. Should be between 0 and 1.
    salt_vs_pepper (float): Proportion of salt noise vs. pepper noise. Should be between 0 and 1.

    Returns:
    numpy.ndarray: Noisy image.
    """
    # Make a copy of the image to avoid modifying the original
    noisy_image = np.copy(image)

    # Calculate the number of pixels to alter
    total_pixels = image.size
    num_salt = np.ceil(amount * total_pixels * salt_vs_pepper).astype(int)
    num_pepper = np.ceil(amount * total_pixels * (1.0 - salt_vs_pepper)).astype(int)

    # Generate random coordinates for salt noise
    coords_salt = tuple([ np.random.randint(0, i - 1, num_salt) for i in image.shape ])
    noisy_image[coords_salt] = image.max()

    # Generate random coordinates for pepper noise
    coords_pepper = tuple([ np.random.randint(0, i - 1, num_pepper) for i in image.shape ])
    noisy_image[coords_pepper] = image.min()

    return noisy_image

# traffic_big = video_to_greyscale_array('traffic_highway.mp4', (1920,1080))
# traffic = traffic_big[:25,600:,500:1250]
# traffic2 = traffic[3:23,100:300,175:375]
# traffic2_noisy = add_bugs_noise(traffic2, maxradius=10, meannumber=3, meantime=1)




