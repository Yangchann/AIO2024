import cv2
import os
import numpy as np


def l1_distance(x, y):
    return abs(x - y)


def l2_distance(x, y):
    return (x - y) ** 2


def find_disparity(left, right, x, y, disparity_range, distance_measure=l1_distance):
    disparity = 0
    cost_min = 255

    for j in range(disparity_range):
        cost = 255
        if (x - j >= 0):
            if distance_measure == l1_distance:
                cost = l1_distance(int(left[y, x]), int(right[y, x-j]))
            elif distance_measure == l2_distance:
                cost = l2_distance(int(left[y, x]), int(right[y, x-j]))

        if cost < cost_min:
            cost_min = cost
            disparity = j

    return disparity


def pixel_wise_matching_l1(left_img, right_img, disparity_range):
    left = cv2.imread(left_img, 0)
    right = cv2.imread(right_img, 0)

    left = left.astype(np.float32)
    right = right.astype(np.float32)

    height, width = left.shape[:2]

    depth = np.zeros((height, width), np.uint8)
    scale = 16

    for y in range(height):
        for x in range(width):
            disparity = find_disparity(
                left, right, x, y, disparity_range, distance_measure=l1_distance)
            depth[y, x] = disparity * scale

    return depth


def pixel_wise_matching_l2(left_img, right_img, disparity_range):
    left = cv2.imread(left_img, 0)
    right = cv2.imread(right_img, 0)

    left = left.astype(np.float32)
    right = right.astype(np.float32)

    height, width = left.shape[:2]

    depth = np.zeros((height, width), np.uint8)
    scale = 16

    for y in range(height):
        for x in range(width):
            disparity = find_disparity(
                left, right, x, y, disparity_range, distance_measure=l2_distance)
            depth[y, x] = disparity * scale

    return depth


def pixel_wise_matching_with_slicing(left_img, right_img, disparity_range):
    left = cv2.imread(left_img, 0)
    right = cv2.imread(right_img, 0)

    left = left.astype(np.float32)
    right = right.astype(np.float32)

    height, width = left.shape[:2]
    scale = 16

    cost = np.full((height, width, disparity_range), 255, np.uint8)
    for j in range(disparity_range):
        left_d = left[:, j:width]
        right_d = right[:, 0:width-j]

        # You can change the distance measure here
        cost[:, j:width, j] = l1_distance(left_d, right_d)

    min_cost = np.argmin(cost, axis=2)
    depth = min_cost * scale
    depth = depth.astype(np.uint8)

    return depth


def save_result(depth_img, results_path, method_name, file_name):
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    print("Saving results...")
    cv2.imwrite(os.path.join(
        results_path, method_name + "_" + file_name), depth_img)
    cv2.imwrite(os.path.join(results_path, method_name + "_color_" +
                file_name), cv2.applyColorMap(depth_img, cv2.COLORMAP_JET))
    print("Done.")


if __name__ == "__main__":
    # Create the results directory
    if not os.path.exists("./results"):
        os.makedirs("./results")
    results_path = r"./results/pixel_wise_matching/"

    left_img_path = r"./data/tsukuba/left.png"
    right_img_path = r"./data/tsukuba/right.png"
    disparity_range = 16

    pixel_wise_result_l1 = pixel_wise_matching_l1(
        left_img=left_img_path,
        right_img=right_img_path,
        disparity_range=disparity_range
    )
    save_result(pixel_wise_result_l1, results_path, "L1", "tsukuba.png")

    pixel_wise_result_l2 = pixel_wise_matching_l2(
        left_img=left_img_path,
        right_img=right_img_path,
        disparity_range=disparity_range
    )
    save_result(pixel_wise_result_l2, results_path, "L2", "tsukuba.png")

    pixel_wise_result_with_slicing = pixel_wise_matching_with_slicing(
        left_img=left_img_path,
        right_img=right_img_path,
        disparity_range=disparity_range
    )

    print("Difference of pixel-wise matching without slicing and with slicing:",
          np.sum(pixel_wise_result_with_slicing - pixel_wise_result_l1))
