import matplotlib.pyplot as plt
import numpy as np

img = plt.imread("pink_lake.png")
plt.imshow(img)
# plt.show()
img_add = img

# Adding a constant value of 0.25 to each pixel then clipping the img to keep the values inside [0, 1]
img_add = np.add(0.25, img)
img_add = np.clip(img_add, 0, 1)
plt.imsave('img_add.png', img_add)

# Creating a zero matrix with the dimensions of the image
img_chan_0 = np.zeros(img.shape)
img_chan_1 = np.zeros(img.shape)
img_chan_2 = np.zeros(img.shape)

# Passing in the color channels
img_chan_0[:, :, 0] = img[:, :, 0]
img_chan_1[:, :, 1] = img[:, :, 1]
img_chan_2[:, :, 2] = img[:, :, 2]
plt.imsave('img_chan_0.png', img_chan_0)
plt.imsave('img_chan_1.png', img_chan_1)
plt.imsave('img_chan_2.png', img_chan_2)

# Creating a zero matrix with the dimensions of the image
img_gray = np.zeros(img.shape)

# Defining the equation to make a pixel grey scale
p = 0.299*img[:, :, 0] + 0.587*img[:, :, 1] + 0.114*img[:, :, 2]
img_gray[:, :, 0] = p
img_gray[:, :, 1] = p
img_gray[:, :, 2] = p

plt.imsave('img_gray.png', img_gray)

# Cropping the image by dividing its height by half
x = int(img.shape[0]/2)
y = img.shape[1]
img_crop = img[:x, :y, :]
plt.imsave('img_crop.png', img_crop)

# Flipping the image by reversing its values along the y axis vertically
img_flip_vert = np.zeros(img.shape)
for i in range(0, img.shape[0]):
    for j in range(0, img.shape[1]):
        img_flip_vert[img.shape[0] - 1 - i][j] = img[i][j]

plt.imsave('img_flip_vert.png', img_flip_vert)