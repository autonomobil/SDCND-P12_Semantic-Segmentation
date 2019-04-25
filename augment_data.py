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
Y_image_link = []  # glob.glob('./data/data_road/training/gt_image_2/u*.png')
flip = True

if flip:
    # Flip Vertically, save normalized old and normalized flipped version
    for i in range(len(X_images_link)):
        X_image_link = X_images_link[i]
        base_name_X = os.path.basename(X_image_link)
        base_name_X = os.path.splitext(base_name_X)[0]

        Y_image_link = X_image_link.replace("_0", "_road_0")
        Y_image_link = Y_image_link.replace("image_2", "gt_image_2")
        base_name_Y = os.path.basename(Y_image_link)
        base_name_Y = os.path.splitext(base_name_Y)[0]

        old_img_X = imageio.imread(X_image_link)
        old_img_Y = imageio.imread(Y_image_link)

        new_img_X = cv2.flip(old_img_X, 1)
        new_img_Y = cv2.flip(old_img_Y, 1)

        old_img_X = helper2.normalize_img(old_img_X)
        new_img_X = helper2.normalize_img(new_img_X)

        # BGR 2 RGB 
        old_img_X = old_img_X [..., ::-1]
        new_img_X = new_img_X [..., ::-1]
        
        old_img_Y = old_img_Y [..., ::-1]
        new_img_Y = new_img_Y [..., ::-1]

        # old_img_X = cv2.cvtColor(old_img_X, cv2.COLOR_BGR2RGB)
        # new_img_X = cv2.cvtColor(new_img_X, cv2.COLOR_BGR2RGB)
        # old_img_Y = cv2.cvtColor(old_img_Y, cv2.COLOR_BGR2RGB)
        # new_img_Y = cv2.cvtColor(new_img_Y, cv2.COLOR_BGR2RGB)

        cv2.imwrite("./data/data_road/training/image_2/" +
                    base_name_X + "1.png", new_img_X)
        cv2.imwrite("./data/data_road/training/gt_image_2/" +
                    base_name_Y + "1.png", new_img_Y)

        cv2.imwrite("./data/data_road/training/image_2/" +
                    base_name_X + ".png", old_img_X)
        cv2.imwrite("./data/data_road/training/gt_image_2/" +
                    base_name_Y + ".png", old_img_Y)

# Augment images
#           colorshift, zoom, rotate, move, warp, noise
aug_ranges = [0.55, 0.12, 5, 5, 5, 0.001]

for i in range(len(X_images_link)):
    X_image_link = X_images_link[i]
    base_name_X = os.path.basename(X_image_link)
    base_name_X = os.path.splitext(base_name_X)[0]
    Y_image_link = X_image_link.replace("_0", "_road_0")
    Y_image_link = Y_image_link.replace("image_2", "gt_image_2")

    base_name_Y = os.path.basename(Y_image_link)
    base_name_Y = os.path.splitext(base_name_Y)[0]

    old_img_X = imageio.imread(X_image_link)
    old_img_Y = imageio.imread(Y_image_link)

    for no_augmen in range(3):

        old_img_X = helper2.normalize_img(old_img_X)
        old_img_Y = helper2.normalize_img(old_img_Y)

        new_img_X, new_img_Y = helper2.augmen_img(
            old_img_X, old_img_Y, aug_ranges, plot=0)

        new_img_X = helper2.normalize_img(new_img_X)
        new_img_Y = helper2.normalize_img(new_img_Y)

        new_img_X = new_img_X [..., ::-1]
        new_img_Y = new_img_Y [..., ::-1]
        # new_img_X = cv2.cvtColor(new_img_X, cv2.COLOR_BGR2RGB)
        # new_img_Y = cv2.cvtColor(new_img_Y, cv2.COLOR_BGR2RGB)

        cv2.imwrite("./data/data_road/training/image_2/" +
                    base_name_X + "0{}.png".format(no_augmen),
                    new_img_X)
        cv2.imwrite("./data/data_road/training/gt_image_2/" +
                    base_name_Y + "0{}.png".format(no_augmen),
                    new_img_Y)
