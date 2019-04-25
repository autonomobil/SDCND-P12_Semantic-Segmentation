import os
import numpy as np
import random
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.colors as mplcolors
import cv2
import scipy.misc
import sys



def normalize_img(image, range=[0, 255]):
    normalizedImg = np.zeros(image.shape[:2])
    normalizedImg = cv2.normalize(
        image,  normalizedImg, range[0], range[1], cv2.NORM_MINMAX)
    return normalizedImg


def noise_generator(noise_type, image, amount=0.001):
    # Source of the code is based on an excelent piece code from stackoverflow
    # http://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv
    """
    Generate noise to a given Image based on required noise type

    Input parameters:
        image: ndarray (input image data. It will be converted to float)

        noise_type: string
            'gauss'        Gaussian-distrituion based noise
            'poission'     Poission-distribution based noise
            's&p'          Salt and Pepper noise, 0 or 1
            'speckle'      Multiplicative noise using out = image + n*image
                           where n is uniform noise with specified mean & variance
    """
    row, col, ch = image.shape
    if noise_type == "gauss":
        mean = 0.0
        var = amount*5
        sigma = var**0.5
        gauss = np.array(image.shape)
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss
        return noisy.astype('uint8')
    elif noise_type == "s&p":
        s_vs_p = 0.5
        # amount = 0.004
        out = image
        # Generate Salt '1' noise
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
        out[coords] = 255
        # Generate Pepper '0' noise
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
        out[coords] = 0
        return out
    elif noise_type == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_type == "speckle":
        gauss = np.random.randn(row, col, ch)
        gauss = gauss.reshape(row, col, ch)
        noisy = image + image * gauss
        return noisy
    else:
        return image


def augmen_img(old_img_X, old_img_Y, aug_ranges=[0.5, 0.08, 8, 2, 3, 0.004], plot=0):

    img_X = old_img_X.copy()
    img_Y = old_img_Y.copy()
    y, x = img_X.shape[:2]

    img_Y = img_Y.astype('float64')
    img_Y = img_Y.astype('float64')

    range_colorshift = aug_ranges[0]
    range_zoom = aug_ranges[1]
    range_rotate = aug_ranges[2]
    range_move = aug_ranges[3]
    warp_factor = aug_ranges[4]
    noise_amount = aug_ranges[4]

    # COLORSHIFT
    if range_colorshift is not 0:

        for color in range(3):
            rand_shift = np.random.uniform(1 - range_colorshift, 1)
            img_X[:, :, color] = img_X[:, :, color] * rand_shift

        # img_X = img_X.astype(np.uint16)

    # WARP
    if warp_factor is not 0:
        f1 = random.uniform(-1, 1)
        f2 = random.uniform(-1, 1)
        f3 = random.uniform(-1, 1)

        pts1 = np.float32([[0, 0], [y, 0], [0, x]])
        pts2 = np.float32([[0 + warp_factor*f1, 0 + warp_factor*f1], [y - warp_factor *
                                                                      f2, 0 + warp_factor*f2], [0+warp_factor*f3, x-warp_factor*f3]])

        M = cv2.getAffineTransform(pts1, pts2)
        img_X = cv2.warpAffine(img_X, M, (x, y))
        img_Y = cv2.warpAffine(img_Y, M, (x, y))

    # ZOOM
    if range_zoom is not 0:
        zoom_factor = random.uniform(1 - range_zoom, 1 + range_zoom)

        newx = (x // zoom_factor)
        deltax = abs(newx - x)
        newy = (y // zoom_factor)
        deltay = abs(newy - y)

        pts1 = np.float32([[0, 0], [x, 0], [0, y]])
        pts2 = np.float32(
            [[deltax, deltay], [newx+deltax, deltay], [deltax, newy+deltay]])

        M = cv2.getAffineTransform(pts1, pts2)
        img_X = cv2.warpAffine(img_X, M, (x, y))
        img_Y = cv2.warpAffine(img_Y, M, (x, y))

    # ROTATE
    if range_rotate is not 0:
        angle = np.random.randint(-range_rotate, range_rotate)

        M = cv2.getRotationMatrix2D((x/2, y/2), angle, 1)
        img_X = cv2.warpAffine(img_X, M, (x, y))
        img_Y = cv2.warpAffine(img_Y, M, (x, y))

    # MOVE
    if range_move is not 0:
        dx = np.random.randint(-range_move, range_move)
        dy = np.random.randint(-range_move, range_move)

        # roll
        img_X = np.roll(img_X, dx, 1)
        img_X = np.roll(img_X, dy, 0)
        img_Y = np.roll(img_Y, dx, 1)
        img_Y = np.roll(img_Y, dy, 0)

        if dx > 0:
            img_X[:, 0:dx, :] = 0
            img_Y[:, 0:dx, :] = 0

        elif dx < 0:
            img_X[:, dx:, :] = 0
            img_Y[:, dx:, :] = 0

        if dy > 0:
            img_X[0:dy, :, :] = 0
            img_Y[0:dy, :, :] = 0

        elif dy < 0:
            img_X[dy:, :, :] = 0
            img_Y[dy:, :, :] = 0

    # NOISE
    if noise_amount is not 0:
        img_X = noise_generator("s&p", img_X)

    flags = np.any(img_Y != [255., 0., 255.], axis=-1)
    img_Y[flags] = [255.0, 0., 0.]

    img_Y = img_Y.astype(np.int16)
    img_Y = img_Y.astype(np.int16)

    if plot:
        fig, axs = plt.subplots(2, 2, figsize=(15, 15))
        axs[0, 0].imshow(old_img_X)
        axs[0, 0].set_title('original')

        axs[1, 0].imshow(img_X)
        axs[1, 0].set_title('augmented')

        axs[0, 1].imshow(old_img_Y)
        axs[0, 1].set_title('original')

        axs[1, 1].imshow(img_Y)
        axs[1, 1].set_title('augmented')
        plt.show()

    return img_X, img_Y


# The MIT License (MIT)
# Copyright (c) 2016 Vladimir Ignatev
#
# Permission is hereby granted, free of charge, to any person obtaining 
# a copy of this software and associated documentation files (the "Software"), 
# to deal in the Software without restriction, including without limitation 
# the rights to use, copy, modify, merge, publish, distribute, sublicense, 
# and/or sell copies of the Software, and to permit persons to whom the Software 
# is furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included 
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT
# OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE 
# OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()  # As suggested by Rom Ruben (see: http://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console/27871113#comment50529068_27871113)