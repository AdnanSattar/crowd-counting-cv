import glob
import math
import os
import sys
import time

import cv2
import h5py
import numpy as np
import scipy.io
import scipy.io as io
import scipy.spatial
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter

"""please set your dataset path"""
root = "./dataset/ShanghaiTech/ShanghaiTech/"

part_A_test = os.path.join(root, "part_A/test_data", "images")
path_sets = [part_A_test]

if not os.path.exists(part_A_test):
    sys.exit("The path is wrong, please check the dataset path.")


img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, "*.jpg")):
        img_paths.append(img_path)

img_paths.sort()

f = open("./A_gt.txt", "w+")
k = 1
for img_path in img_paths:

    print(img_path)
    mat = io.loadmat(
        img_path.replace(".jpg", ".mat")
        .replace("images", "ground_truth")
        .replace("IMG_", "GT_IMG_")
    )
    Gt_data = mat["image_info"][0][0][0][0][0]
    f.write("{} {} ".format(k, len(Gt_data)))

    for data in Gt_data:

        sigma_s = 4
        sigma_l = 8
        f.write(
            "{} {} {} {} {} ".format(
                math.floor(data[0]), math.floor(data[1]), sigma_s, sigma_l, 1
            )
        )
    f.write("\n")

    k = k + 1
f.close()
