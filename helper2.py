import os
import numpy as np
import random
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.colors as mplcolors
import cv2
import helper2


def normalize_img(image, option=0):

    image = image.astype('float64')

    for color in range(3):
        min_val = image[:, :, color].min()
        min_val = min_val.astype('float64')
        max_val = image[:, :, color].max()
        max_val = max_val.astype('float64')

        val_range = (max_val - min_val)

        ###### Normalize for range 0, 1
        if option == 0:
            image[:, :, color] = (
                image[:, :, color] - (min_val)) / (val_range)
        ###### Normalize for range -1, 1
        elif option == 1:
            image[:, :, color] = (
                image[:, :, color] - (val_range/2 + min_val)) / (val_range/2)

    return image


def augmen_img(old_img_X, old_img_Y, aug_ranges=[0.5, 0.08, 8, 2, 3], plot=0):

    img_X = old_img_X.copy()
    img_Y = old_img_Y.copy()
    y, x = img_X.shape[:2]

    range_colorshift = aug_ranges[0]
    range_zoom = aug_ranges[1]
    range_rotate = aug_ranges[2]
    range_move = aug_ranges[3]
    warp_factor = aug_ranges[4]

    ######################## COLORSHIFT
    if range_colorshift is not 0:
        
        # img_X = img_X.astype('float64')
        
        for color in range(3):

            rand_shift = np.random.uniform(1 - range_colorshift, 1)

            img_X[:, :, color] = img_X[:, :, color] * rand_shift
        
        # img_X = img_X.astype(np.uint16)
        

    ######################## WARP
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

    ######################## ZOOM
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

    ########################## ROTATE
    if range_rotate is not 0:
        angle = np.random.randint(-range_rotate, range_rotate)

        M = cv2.getRotationMatrix2D((x/2, y/2), angle, 1)
        img_X = cv2.warpAffine(img_X, M, (x, y))
        img_Y = cv2.warpAffine(img_Y, M, (x, y))

    ####################### MOVE
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

    flags = np.any(img_Y != [1., 0., 1.], axis=-1)
    img_Y[flags] = [1.0, 0., 0.]

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
