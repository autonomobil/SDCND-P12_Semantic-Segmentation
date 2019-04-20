import os
import re
import numpy as np
import random
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.colors as mplcolors
import cv2
import helper2
import imageio

X_images_link = glob.glob("./data/data_road/training/image_2/u*.png")
Y_images_link = []  # glob.glob('./data/data_road/training/gt_image_2/u*.png')

#           colorshift, zoom, rotate, move, warp
aug_ranges = [0.55, 0.12, 4, 4, 4]

for i in range(len(X_images_link)):
        base_name_X = os.path.basename(X_images_link[i])
        Y_images_link.append(X_images_link[i].replace("_0", "_road_0"))
        Y_images_link[i] = Y_images_link[i].replace("image_2", "gt_image_2")
        
        old_img_X = mpimg.imread(X_images_link[i])
        old_img_Y = mpimg.imread(Y_images_link[i])
        base_name_Y = os.path.basename(Y_images_link[i])

        base_name_X = os.path.splitext(base_name_X)[0]
        base_name_Y = os.path.splitext(base_name_Y)[0]

        for no_augmen in range(3):

                old_img_X = helper2.normalize_img(old_img_X)
                new_img_X, new_img_Y = helper2.augmen_img(old_img_X, old_img_Y, aug_ranges, plot=0)

                new_img_X = helper2.normalize_img(new_img_X)

                new_img_X = 255 * new_img_X  # Now scale by 255
                new_img_X = new_img_X.astype(np.uint8)

                new_img_Y = 255 * new_img_Y  # Now scale by 255
                new_img_Y = new_img_Y.astype(np.uint8)

                imageio.imwrite("./data/data_road/training/image_2/" + base_name_X + "{}.png".format(no_augmen), new_img_X)
                imageio.imwrite("./data/data_road/training/gt_image_2/" + base_name_Y + "{}.png".format(no_augmen), new_img_Y)

    # X_train.append(new_images[0])
    # Y_train.append(new_images[1])
