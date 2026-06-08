import numpy as np
import matplotlib.pyplot as plt

def plot_u_light(u, title=None, save=False, name='newname.jpg', vmin=None, vmax=None, normalize=False, border_width=2):
    '''
    Plots a 4-D array as a single image with borders between cells.
    Much faster than creating multiple subplots.
    
    Parameters:
    - u: 4D numpy array of shape (M1, M2, N1, N2)
    - title: Plot title
    - save: Whether to save the figure
    - name: Filename if saving
    - vmin, vmax: Value range for display
    - normalize: Whether to normalize to min/max of u
    - border_width: Width of borders between cells in pixels
    '''
    
    if vmax is None and normalize: vmax = np.max(u)
    if vmin is None and normalize: vmin = np.min(u)
    
    M1, M2, N1, N2 = u.shape
    
    # Create single image with borders
    img_height = M1 * N1 + (M1 - 1) * border_width
    img_width = M2 * N2 + (M2 - 1) * border_width
    
    # Initialize with border color (white for grayscale)
    combined_img = np.ones((img_height, img_width)) if vmax is None else np.ones((img_height, img_width)) * vmax
    
    # Fill in each cell
    for i in range(M1):
        for j in range(M2):
            row_start = i * (N1 + border_width)
            row_end = row_start + N1
            col_start = j * (N2 + border_width)
            col_end = col_start + N2
            
            combined_img[row_start:row_end, col_start:col_end] = u[i, j]
    
    # Plot the combined image
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(combined_img, cmap='gray', vmin=vmin, vmax=vmax)
    ax.axis('off')
    
    if title:
        plt.title(title, fontsize=14)
    
    if save:
        plt.savefig(name, dpi=200, bbox_inches='tight')
    
    plt.tight_layout()
    plt.show()

def plot_u(u, ndim_y, title=None, save=False, name='newname.jpg', return_fig=False, vmin=None, vmax=None, normalize=False, scale=1):
    '''u is a 4-D array, and it plots it as a table of plt.imshow(u[i,j]) subplots'''
    
    if ndim_y == 3:
        plot_u_3d(u)
        return
    
    if vmax in {None} and normalize in {True}: vmax=np.max(u)
    if vmin in {None} and normalize in {True}: vmin=np.min(u)
    
    M1, M2 = u.shape[0], u.shape[1]
    
    # Create a figure with a grid of subplots
    fig, axes = plt.subplots(M1, M2, figsize=(scale*M2, scale*M1) )

    # If M1 or M2 is 1, axes may not be a 2D array; ensure it is
    if M1 == 1: axes = np.expand_dims(axes, axis=0)
    if M2 == 1: axes = np.expand_dims(axes, axis=1)
    
    # Loop over each image and display it
    for i in range(M1):
        for j in range(M2):
            axes[i, j].imshow(u[i, j], cmap='gray', vmin=vmin, vmax=vmax)
            axes[i, j].axis('off')  # Optional: turn off axis labels
    
            
    plt.suptitle(title)
    if save==True: plt.savefig(name, dpi=200)

    plt.tight_layout() # Adjust layout and display the plot (optional)
    
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
    

def save_u_asgif(*arrays, titles=None, filename='animation.gif'):
    
    from imageio import v2 as imageio
    import io
    
    ver, hor = arrays[0].shape[:2]
    
    images = []
    if titles is None: titles = len(arrays) * [''] # if titles is None, I assign all empty titles
    for u, title in zip(arrays, titles):
        fig, axs = plt.subplots(ver, hor)
        if ver == 1: axs = axs.reshape(1,hor)
        if hor == 1: axs = axs.reshape(ver,1)
        for i in range(ver):
            for j in range(hor):
                axs[i,j].imshow(u[i,j], cmap='gray')
                axs[i,j].set_xticks([])
                axs[i,j].set_yticks([])
        plt.suptitle(title)
        # Save the figure to a buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        # Read the image from the buffer
        image = imageio.imread(buf)
        images.append(image)
        buf.close()
        plt.close(fig)  # Close the figure to free up memory
    
    # Create a GIF from the images
    imageio.mimsave(filename, images, duration=0.5)

def savecurve_asgif(*data, start=0, stop=None, interval=400, filename='curve.gif'):
    import matplotlib.animation as animation

    data = list(data) # data[i] is meant to be a curve (TxNxN shape)
    n = len(data)
    
    for k, x in enumerate(data):
        if isinstance(x, dict):
            first_key = list(x.keys())[0]
            data[k] = x[first_key]
    
    if stop == None: stop = len(data[0])
    
    fig, axes = plt.subplots(1, n, figsize=(1.5*n, 1.5))
    if n == 1: axes = [axes]
    list_of_AxesImage = []
    for k, ax in enumerate(axes):
        list_of_AxesImage += [ax.imshow(data[k][start], cmap='gray', animated=True)]
    
    def update(frame):
        for k, ax in enumerate(list_of_AxesImage):
            ax.set_array(data[k][frame])
        return list_of_AxesImage
    
    ani = animation.FuncAnimation(fig, update, frames=range(start, stop), interval=interval, blit=True)
    ani.save(filename, writer='pillow')
    plt.close(fig)  # Close the figure to prevent display in notebooks
    print(f"GIF saved as {filename}")
    return

def plotcurve_asgif(*data, start=0, stop=None, filename=None, interval=1000):
    delete = False
    if filename == None:
        import os
        filename, delete = 'temp.gif', True
        
    savecurve_asgif(*data, start=start, stop=stop, interval=interval, filename=filename)
    plot_fromgif(gif_path=filename, window_title='Animated GIF', delay=250)
    
    if delete and os.path.exists(filename):
        os.remove(filename)
        print(f"Deleted file: {filename}")
    plt.close()
    return

def plot_fromgif(gif_path='animation.gif', window_title='Animated GIF', delay=250):
    import tkinter as tk
    from PIL import Image, ImageTk, ImageSequence

    # Initialize the Tkinter root window
    root = tk.Tk()
    root.title(window_title)
    
    try:
        frames = [ImageTk.PhotoImage(img)
                  for img in ImageSequence.Iterator(Image.open(gif_path))] # Load the GIF frames
    except Exception as e:
        print(f"Error loading GIF: {e}")
        root.destroy()  # Destroy the root window before exiting
        return

    # Define the update function to animate the GIF
    def update(ind):
        frame = frames[ind % len(frames)]
        label.configure(image=frame)
        root.after(delay, update, ind+1)

    # Create and pack the label widget
    label = tk.Label(root)
    label.pack()

    # Start the animation
    root.after(0, update, 0)
    root.mainloop()
    return

def printcurve(*data, start=0, stop=None):
    '''prints a curve; it can also handle the output of Chambolle-Pock (which is a dictionary)'''
    n = len(data)
    data = list(data)
    for i, x in enumerate(data):
        if type(x) == dict:
            first_key = list(x.keys())[0]
            data[i] = x[first_key]
    if stop == None: stop = len(data[0])
    for t in range(start,stop):
        fig, axes = plt.subplots(1, n, figsize=(5*n, 5))
        if n == 1: axes = [axes]
        for k, ax in enumerate(axes):
            ax.imshow(data[k][t], cmap='gray')
            ax.text(0.5, 0.5, f"Max\n{np.max(data[k][t]):.2f}", ha='center', va='center', fontsize=12)
        plt.show()
    return
