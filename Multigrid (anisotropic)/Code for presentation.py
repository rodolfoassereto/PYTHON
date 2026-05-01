''' Code for the presentatuion '''
import numpy as np
import matplotlib.pyplot as plt
import gradients_and_TV_multigrid as tv
import prox_FF as pf
import prox_GG as pg
import building_data as bd
import multiscale_DR as DR

biggest_size = 9
k = 5
noiselevel = 0.1

im_original = bd.import_image('abstract_pattern.jpg', biggest_size, normalize=True)
im_noisy = bd.add_gaussian_noise(im_original, sig=noiselevel)

plt.imshow(im_original, cmap='gray', vmin=0, vmax=1)
plt.show()
plt.imshow(im_noisy, cmap='gray', vmin=0, vmax=1)
plt.show()

list_of_images = [im_noisy]
for i in range(k):
    list_of_images = list_of_images + [bd.halve(list_of_images[-1])]

def plotsubplots():
    fig = plt.figure()
    sub = fig.add_subplot(1,k,1)
    sub.set_title('0')
    sub.imshow(list_of_images[0], cmap='gray')
    sub.set_title(f"TV = 1575")
    plt.axis('off')
    for i in range(2,k):
        sub = fig.add_subplot(1,k,i)
        sub.set_title(f"TV = {2**i * int(tv.TV(list_of_images[i]))}")
        sub.imshow(list_of_images[i], cmap='gray')
        plt.axis('off')
    plt.savefig('my_plot.png', dpi=300)
    plt.show()
    return

# PLOTTING NOISY IMAGE
plt.figure(dpi=250)
plt.imshow(im_noisy, cmap='gray', vmin=0, vmax=1)
plt.title('noisy image')
plt.show()

for i, x in enumerate(list_of_images):
    print(2**i * tv.TV(x))



# PLOTTING TVs COMPARISON

# fig = plt.figure()
# sub = fig.add_subplot(1,k,1)
# sub.set_title(r"TV = 1563")
# sub.imshow(list_of_images[0], cmap='gray')
# plt.axis('off')
# sub = fig.add_subplot(1,k,2)
# sub.set_title(r"2TV = 558")
# sub.imshow(list_of_images[2], cmap='gray')
# plt.axis('off')
# sub = fig.add_subplot(1,k,3)
# sub.set_title(r"2$^2$TV = 308")
# sub.imshow(list_of_images[3], cmap='gray')
# plt.axis('off')
# sub = fig.add_subplot(1,k,4)
# sub.set_title(r"2$^3$TV = 208")
# sub.imshow(list_of_images[4], cmap='gray')
# plt.axis('off')
# plt.savefig('my_plot.png', dpi=300)
# plt.show()

# for i in range(len(list_of_images)):
#     print(list_of_images[i].shape)
#     print(tv.TV(list_of_images[i]))



# zer = np.zeros(6)
# mymat = np.zeros((6,6))
# mymat[0,2] = 0.25
# mymat[1:3,1] = 0.25
# mymat[2,0] = 1
# mymat[2,1] = 0.25
# print(mymat)
# print(tv.TV(mymat))
# print(2*tv.TV(bd.halve(mymat)))

