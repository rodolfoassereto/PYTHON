from PIL import Image
import numpy as np

def import_image(filepath, target_shape=None, normalize=True):
    """
    Import an image and convert it to a grayscale matrix.
    
    Parameters:
    -----------
    filepath : str
        Path to the image file (absolute path from C:/ or relative path from working directory)
    target_shape : tuple of int, optional
        Desired (height, width) for cropping/resizing. If None, keeps original size.
    normalize : bool, optional
        If True, normalizes pixel values to [0, 1]. Default is True.
    
    Returns:
    --------
    img : numpy.ndarray
        2D array representing the grayscale image
    """
    
    # Load image
    img = Image.open(filepath)
    
    # Convert to grayscale
    img = img.convert('L')
    
    # Crop/resize if target_shape is specified
    if target_shape is not None:
        # Resize to target shape (height, width)
        img = img.resize((target_shape[1], target_shape[0]), Image.BILINEAR)
    
    # Convert to numpy array
    img_array = np.array(img, dtype=np.float64)
    
    # Normalize if requested
    if normalize:
        img_min = np.min(img_array)
        img_max = np.max(img_array)
        
        if img_max > img_min:
            img_array = (img_array - img_min) / (img_max - img_min)
        else:
            img_array = np.zeros_like(img_array)
    
    return img_array