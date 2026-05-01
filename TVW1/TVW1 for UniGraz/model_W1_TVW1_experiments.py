import numpy as np
import matplotlib.pyplot as plt
import sys
# Add Libraries folder to sys.path
libraries_path = r"C:\Users\rodol\My Drive\PYHTON\Libraries"
if libraries_path not in sys.path:
    sys.path.insert(0, libraries_path)
from plottings import plot_u, plot_u_light  # noqa: E402


def create_2d_ring(shape_y, plotit=False):

    # Create a black image with the specified shape
    img = np.zeros(shape_y)

    # Create coordinate grids based on actual shape
    y, x = np.ogrid[:shape_y[0], :shape_y[1]]

    # Define circle parameters (larger circle at bottom)
    # Scale centers and radii based on image size
    circle_center_x = shape_y[1] // 2
    circle_center_y = int(shape_y[0] * 0.5)
    circle_radius = min(shape_y) // 2.25

    # Define ball parameters (smaller circle on top)
    ball_center_x = shape_y[1] // 2
    ball_center_y = int(shape_y[0] * 0.1)
    ball_radius = min(shape_y) // 10

    # Create distance arrays
    circle_dist = np.sqrt((x - circle_center_x)**2 + (y - circle_center_y)**2)
    ball_dist = np.sqrt((x - ball_center_x)**2 + (y - ball_center_y)**2)

    # Create 1-pixel wide ring (perimeter only)
    # Points where distance is within 0.5 pixels of the radius
    ring_mask = np.abs(circle_dist - circle_radius) <= 0.5
    
    # Create filled ball
    ball_mask = ball_dist <= ball_radius

    # Set white (1) for the ring and ball
    img[ring_mask] = 1
    img[ball_mask] = 1

    # Display the image if requested
    if plotit:
        plt.figure(figsize=(6, 6))
        plt.imshow(img, cmap='gray', vmin=0, vmax=1)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    return img

def rotate_2d_image(img, phi):
    from scipy.ndimage import rotate
    
    # Rotate the image by phi degrees
    # reshape=False keeps the output shape same as input
    # order=1 uses bilinear interpolation
    # mode='constant' fills empty space with 0 (black)
    rotated = rotate(img, np.degrees(phi), reshape=False, order=1, mode='constant', cval=0)
    
    return rotated

from model_W1_TVW1 import model_W1_TVW1
# %%

from test_models_with_gaussian_mixture import gaussian_mixture, shape_x, shape_y, shape_xy, ndim_x, ndim_y
u_input = gaussian_mixture.copy()
for idx in np.ndindex(shape_x):
     u_input[idx] = gaussian_mixture[idx] / np.sum( gaussian_mixture[idx] )
plot_u(u_input, ndim_y, title="gaussian_mixture")

# %% "4d ring" experiment

shape_ring_y = (64,64)
ndim_ring_y = len(shape_ring_y)
shape_ring_x = (8,8)
shape_ring_xy = shape_ring_x + shape_ring_y
ring = create_2d_ring( shape_ring_y )

ring_4d = np.zeros( shape_ring_xy )
for idx in np.ndindex(shape_ring_x):
    r = np.random.rand() - 0.5
    phi = np.pi * r * 0.4
    rotated_temp = rotate_2d_image( ring, phi )
    ring_4d[idx] = rotated_temp / np.sum(rotated_temp)
plot_u(ring_4d, shape_ring_y)

ring_4d_reg = model_W1_TVW1(ring_4d, shape_ring_x, shape_ring_y, 1, 3000)
plot_u(ring_4d_reg, ndim_ring_y)

# %% "4d bar" experiment

shape_bar_y = (16,16)
ndim_bar_y = len(shape_bar_y)
shape_bar_x = (10,10)
ndim_bar_x = len(shape_bar_x)
shape_bar_xy = shape_bar_x + shape_bar_y

bar = np.zeros(shape_bar_y)
halfwidth_bar = round( (shape_bar_y[0]/8)/2 )
half_y1 = round(shape_bar_y[1]/2)
bar[ :, half_y1 - halfwidth_bar : half_y1 + halfwidth_bar] = 1

half_x0 = round(shape_bar_x[0]/2)
half_x1 = round(shape_bar_x[1]/2)
initial_phis = np.zeros(shape_bar_x)
initial_phis[half_x0:,:half_x1] = np.pi/6
initial_phis[:half_x0,half_x1:] = np.pi/12
initial_phis[half_x0:,half_x1:] = np.pi/4

bar_4d = np.zeros( shape_bar_xy )
for idx in np.ndindex(shape_bar_x):
    r = np.random.rand() - 0.5
    phi = initial_phis[idx] + np.pi * r * 0.1 * 0
    rotated_temp = rotate_2d_image( bar, phi )
    bar_4d[idx] = rotated_temp / np.sum(rotated_temp)
plot_u(bar_4d, shape_bar_y)

bar_4d_reg = model_W1_TVW1(bar_4d, shape_bar_x, shape_bar_y, 1, 3000)
plot_u(bar_4d_reg, ndim_bar_y)

# %%

def normalize_pixel_values(img, max_value):
    """
    Convert image to standardized integer values in [0, max_value].
    
    Parameters:
    -----------
    img : numpy.ndarray
        Input grayscale image
    max_value : int
        Maximum value for the output range
    
    Returns:
    --------
    img_scaled : numpy.ndarray
        Image with integer values in [0, max_value]
    """
    # Normalize to [0, 1] range
    img_min = np.min(img)
    img_max = np.max(img)
    
    # Handle edge case where image is constant
    if img_max == img_min:
        return np.zeros_like(img, dtype=np.uint8 if max_value <= 255 else np.uint16)
    
    # Scale to [0, 1]
    img_normalized = (img - img_min) / (img_max - img_min)
    
    # Scale to [0, max_value] and convert to integer
    img_scaled = np.round(img_normalized * max_value).astype(int)
    
    return img_scaled

def convert_pixels_into_probabilities(img, max_value):

    img0 = normalize_pixel_values(img, max_value)

    img_prob = np.zeros( img.shape + (max_value+1,) )
    for idx in np.ndindex(img.shape):
        img_prob[idx][img0[idx]] = 1

    return img_prob

def convert_probabilities_into_pixels(img_prob):
    
    shape_img, shape_prob = img_prob.shape[:2], img_prob.shape[2:]
    img = np.zeros(shape_img)
    for idx in np.ndindex(shape_img):
        argmax_location = np.argmax( img_prob[idx] )
        img[idx] = argmax_location
    
    return img

from import_data import import_image

target_shape = (64,64)
max_value = 15

image = import_image(r"C:\Users\rodol\My Drive\PYHTON\TVW1\TVW1 for Malga\image.jpg", target_shape=target_shape)

sigma = 0.25
image_noisy = image +  sigma * np.random.normal(size=target_shape)
plt.imshow(image_noisy, cmap='gray')
plt.show()

image_noisy_prob = convert_pixels_into_probabilities(image_noisy, max_value)

shape_image_prob_x, shape_image_prob_y = target_shape, (max_value+1,)

image_prob_reg = model_W1_TVW1(image_noisy_prob, shape_image_prob_x, shape_image_prob_y, 1, 700)

img_reg_fromW1 = convert_probabilities_into_pixels(image_prob_reg)

plt.close()
plt.imshow(img_reg_fromW1)
plt.show()
