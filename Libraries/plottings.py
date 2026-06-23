import numpy as np
import matplotlib.pyplot as plt

def plot_u(u, ndim_y, title=None, save=False, name='newname.jpg', return_fig=False, vmin=None, vmax=None, normalize=False, border_width=None, border_color='lightblue', scale=1):
    '''
    Plots a 4-D array as a single image with borders between cells.
    Much faster than creating multiple subplots.

    Borders are masked (not data values) and painted in border_color, so they
    stay clearly visible regardless of image content or the vmin/vmax range.

    Parameters:
    - u: 4D numpy array of shape (M1, M2, N1, N2)
    - ndim_y: dimensionality of each cell. If 3, defers to plot_u_3d.
    - title: Plot title
    - save: Whether to save the figure
    - name: Filename if saving
    - return_fig: If True, return the figure instead of showing/closing it
    - vmin, vmax: Value range for display
    - normalize: Whether to normalize to min/max of u
    - border_width: Width of borders between cells in pixels (0 for no borders)
    - border_color: Color of the borders between cells
    - scale: Figure size scaling (figsize = scale*M2 by scale*M1)
    '''

    if ndim_y == 3:
        plot_u_3d(u)
        return

    if vmax is None and normalize: vmax = np.max(u)
    if vmin is None and normalize: vmin = np.min(u)

    M1, M2, N1, N2 = u.shape

    if border_width is None:
        border_width = max( int((N1 + N2) / 20), 1 )

    # Create single image with borders
    img_height = M1 * N1 + (M1 - 1) * border_width
    img_width = M2 * N2 + (M2 - 1) * border_width

    # Initialize with NaN; the gaps stay NaN and are painted as masked pixels
    combined_img = np.full((img_height, img_width), np.nan)

    # Fill in each cell
    for i in range(M1):
        for j in range(M2):
            row_start = i * (N1 + border_width)
            row_end = row_start + N1
            col_start = j * (N2 + border_width)
            col_end = col_start + N2

            combined_img[row_start:row_end, col_start:col_end] = u[i, j]

    # Mask the borders so they render in a fixed, content-independent color
    combined_img = np.ma.masked_invalid(combined_img)
    cmap = plt.get_cmap('gray').copy()
    cmap.set_bad(color=border_color)

    # Plot the combined image
    fig, ax = plt.subplots(figsize=(scale * M2, scale * M1))
    ax.imshow(combined_img, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis('off')

    if title:
        plt.suptitle(title, fontsize=14)

    if save:
        plt.savefig(name, dpi=200, bbox_inches='tight')

    plt.tight_layout()

    if return_fig:
        return fig
    else:
        plt.show()
        plt.close(fig)
    return

def plot_u_3d(u, spacing_along_first_axis=2):
    import napari
    spacing = spacing_along_first_axis
    
    if u.ndim < 6: u = u.reshape( (1,) + u.shape )
    M0, M1, M2 = u.shape[:3]
    N0, N1, N2 = u.shape[3:]
    
    arr_to_plot = np.zeros( (spacing*M0*N0, M1*N1, M2*N2) )
    for i0, i1, i2, j0, j1, j2 in np.ndindex(u.shape):
        arr_to_plot[spacing*i0*N0 + j0, i1*N1 + j1, i2*N2 + j2 ] = u[i0,i1,i2,j0,j1,j2]

    viewer = napari.view_image(arr_to_plot)
    napari.run()
