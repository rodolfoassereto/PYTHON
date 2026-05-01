# def add_sub_directories_to_path(root_folder):
# 	import sys, os
# 	for root, dirs, files in os.walk(root_folder): sys.path.append(root)

# add_sub_directories_to_path('C:/Users/rodol/My Drive/PYHTON')

import numpy as np
import matplotlib.pyplot as plt  # noqa: F401
from numpy.random import rand as rd

# Add Libraries folder to sys.path
import sys
libraries_path = r"C:\Users\rodol\My Drive\PYHTON\Libraries"
if libraries_path not in sys.path:
    sys.path.insert(0, libraries_path)

def ar(*args, shape=None):
    return np.array(args).reshape(shape)

import inspect

def print_withname(var, decimals=2):
    # Get the name of the variable from the caller's local scope
    frame = inspect.currentframe().f_back
    var_name = [name for name, value in frame.f_locals.items() if value is var]

    if var_name:
        print(f"{var_name[0]}: {np.round(var, decimals=decimals)}")
    else:
        print(f"Unknown variable: {var}")

def printarr(arr, decimals=2, entries_to_print=np.inf, scientific_notation=False):
    """
    Prints a NumPy array with each entry rounded to the specified number of decimals.
    
    Parameters:
        arr (np.ndarray): The input array.
        decimals (int): Number of decimal places to round to (default is 2).
        full_output (bool): If True, prints the whole array without truncation.
    """
    np.set_printoptions(threshold=entries_to_print)
    if scientific_notation:
        print(np.vectorize(lambda v: f"{v:.2e}")(arr) )
    else:
        print(np.round(arr, decimals=decimals))



ar45 = ar([2**i for i in range(20)]).reshape(4,5)
ar34 = ar([2**i for i in range(12)]).reshape(3,4)
