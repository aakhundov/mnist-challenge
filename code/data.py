import sys
import struct

import numpy as np
import scipy.ndimage as im


# reading data (up to "limit" samples) from the 
# (unpacked) original  MNIST data files. two separate 
# path to image and label data must be specified
def get_data(image_path, label_path, limit):
    with open(image_path, "rb") as f:
        data = f.read()
        magic, n, rows, cols = struct.unpack_from(">4i", data)
        bytes = struct.unpack_from(">{0}B".format(min(n, limit) * rows * cols), data, offset=16)
        images = np.asarray(bytes).reshape((min(n, limit), rows * cols))

    with open(label_path, "rb") as f:
        data = f.read()
        magic, n = struct.unpack_from(">2i", data)
        bytes = struct.unpack_from(">{0}B".format(min(n, limit)), data, offset=8)
        labels = np.asarray(bytes)

    return images, labels


# deskewing the images to acheive better predictive stability.
# scipy.ndimage.interpolation.affine_transform is used for
# the interpolation of the deskewed images' pixel values
def deskew_images(X):
    images = X / 255.0
    count = len(X)
    summed = np.sum(images, axis=1)
    dim = int(np.sqrt(images.shape[1]))
    images = images.reshape(count, dim, dim)

    gridX, gridY = np.mgrid[:dim, :dim]

    meanX = (np.sum(np.sum(images * gridX, axis=1), axis=1) / summed).reshape(count,1,1)
    meanY = (np.sum(np.sum(images * gridY, axis=1), axis=1) / summed).reshape(count,1,1)

    meanFreeGridX = np.tile(gridX, (count,1,1)) - meanX
    meanFreeGridY = np.tile(gridY, (count,1,1)) - meanY

    varX = (np.sum(np.sum(meanFreeGridX ** 2 * images, axis=1), axis=1) / summed).reshape(count,1,1)
    covXY = (np.sum(np.sum(meanFreeGridX * meanFreeGridY * images, axis=1), axis=1) / summed).reshape(count,1,1)
    alpha = covXY / varX

    center = np.array([dim, dim]) / 2.0
    result = np.zeros_like(X, dtype=np.float64)

    for i in range(count):
        affine = np.array([[1.0, 0.0], [alpha[i,0,0], 1.0]])
        offset = np.array([meanX[i,0,0], meanY[i,0,0]]) - np.dot(affine, center)
        transformed = im.interpolation.affine_transform(images[i], affine, offset=offset).ravel()
        t_min, t_max = transformed.min(), transformed.max()
        result[i,:] = (transformed - t_min) / (t_max - t_min)

    return result * 255.0


# three input pre-processing options: normalization,
# adding a column of ones (bias input), and deskewing
def preprocess_data(data, normalize, add_ones, deskew):
    X, T = data[0], data[1]

    if deskew:
        X = deskew_images(X)
    if normalize:
        X = (X - X.mean(axis=0)) / 255.0
    if add_ones:
        X = np.hstack((np.ones((len(X), 1)), X))

    return X, T


# reading and pre-processing training data ("limit", if defined, specifies the number of samples to read)
def get_training_data(limit=sys.maxint, 
    image_path="train-images.idx3-ubyte", label_path="train-labels.idx1-ubyte", 
    normalize=False, add_ones=False, deskew=False):
    return preprocess_data(get_data(image_path, label_path, limit), normalize, add_ones, deskew)

# reading and pre-processing testing data ("limit", if defined, specifies the number of samples to read)
def get_testing_data(limit=sys.maxint, 
    image_path="t10k-images.idx3-ubyte", label_path="t10k-labels.idx1-ubyte", 
    normalize=False, add_ones=False, deskew=False):
    return preprocess_data(get_data(image_path, label_path, limit), normalize, add_ones, deskew)

