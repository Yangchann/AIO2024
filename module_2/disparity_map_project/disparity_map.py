import numpy as np
import cv2
import matplotlib.pyplot as plt

left_img_path = r"./data/tsukuba/left.png"
right_img_path = r"./data/tsukuba/right.png"
disparity_range = 16

left = cv2.imread(left_img_path, cv2.IMREAD_GRAYSCALE)
right = cv2.imread(right_img_path, cv2.IMREAD_GRAYSCALE)

stereo = cv2.StereoBM_create(numDisparities=disparity_range, blockSize=15)
disparity = stereo.compute(left, right)
plt.imshow(disparity, cmap='jet')
plt.show()
