import cv2
import os
import numpy as np


def cosine_similarity(x, y):
    numerator = np.dot(x, y)
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return numerator / denominator


def window_based_maching(left_img, right_img, disparity_range, kernel_size=5):
    left = cv2.imread(left_img, 0)
    right = cv2.imread(right_img, 0)

    left = left.astype(np.float32)
    right = right.astype(np.float32)

    height, width = left.shape[:2]

    depth = np.zeros((height, width), np.uint8)
    kernel_half = int((kernel_size - 1) / 2)
    scale = 3

    for y in range(kernel_half, height - kernel_half):
        for x in range(kernel_half, width - kernel_half):
            disparity = 0
            cost_optimal = -1

            for j in range(disparity_range):
                d = x - j
                cost = -1
                if (d - kernel_half) > 0:
                    w_p = left[y - kernel_half:y + kernel_half +
                               1, x - kernel_half:x + kernel_half + 1]
                    w_pd = right[y - kernel_half:y + kernel_half +
                                 1, d - kernel_half:d + kernel_half + 1]
                    wp_flat = w_p.flatten()
                    wpd_flat = w_pd.flatten()

                    cost = cosine_similarity(wp_flat, wpd_flat)

                if cost > cost_optimal:
                    cost_optimal = cost
                    disparity = j

            depth[y, x] = disparity * scale

    return depth


def save_results(depth_img, results_path, method_name, file_name):
    if not os.path.isdir(results_path):
        os.mkdir(results_path)
    print("Saving result...")
    cv2.imwrite(os.path.join(
        results_path, method_name + "_" + file_name), depth_img)
    cv2.imwrite(os.path.join(results_path, method_name + "_color_" + file_name),
                cv2.applyColorMap(depth_img, cv2.COLORMAP_JET))
    print("Done.")


if __name__ == "__main__":

    if not os.path.isdir("./results"):
        os.mkdir("./results")
    results_path = r"./results/window_based_matching/"

    left_img_path = r"./data/Aloe_images/Aloe/Aloe_left_1.png"
    right_img_path = r"./data/Aloe_images/Aloe/Aloe_right_2.png"
    disparity_range = 16
    kernel_size = 3

    disparity_map = window_based_maching(
        left_img_path, right_img_path, disparity_range, kernel_size)
    save_results(disparity_map, results_path, "cos", "Aloe2.png")
