# %%
import numpy as np
import matplotlib.pyplot as plt
import sys

# Add Libraries folder to sys.path
libraries_path = r"C:\Users\rodol\My Drive\PYHTON\Libraries"
if libraries_path not in sys.path:
    sys.path.insert(0, libraries_path)
from differential_operators import tv_1d


# Generate random points in 3D TV ball
d = 4
n_points = 10000000
points = np.zeros((n_points,d))
points_random = 3 * (np.random.randn(n_points, d-1) - 0.5)
points[:,1:] = points_random

tv_norms = np.array( [x for x in points if tv_1d(x) <= 1 ] )

# %%

plt.close('all')

# Create 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(tv_norms[:, 1], tv_norms[:, 2], tv_norms[:, 3], alpha=0.5, s=1)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Points in 3D TV Ball')
plt.ion()