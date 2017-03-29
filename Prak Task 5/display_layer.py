import numpy as np
#from matplotlib import pyplot as plt
from math import ceil
#from skimage.io import imshow, imsave
from matplotlib.pyplot import imshow, imsave
from skimage.transform import rescale

def display_layer(X, filename='layer.png'):
    res = convert_to_table(X)
    if filename is None:
        imshow(res)
    else:
        imsave(filename, np.uint8(res * 255))

def convert_to_table(X):
    N = X.shape[0]
    k = ceil(N ** 0.5)
    d = round((X.shape[1] // 3) ** 0.5)
    images = X.reshape(N, d, d, 3)
    res = np.zeros((ceil(N / k) * (d + 1) + 1, k * (d + 1) + 1, 3))
    for i in range(N):
        x = i // k
        y = i % k
        res[x * (d + 1) + 1 : (x + 1) * (d + 1), y * (d + 1) + 1 : (y + 1)  * (d + 1), :] = images[i]
    res = rescale(res, 5, order=0)
    return res