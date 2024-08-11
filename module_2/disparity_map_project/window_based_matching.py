import cv2
import os
import numpy as np


def l1_distance(x, y):
    return abs(x - y)


def l2_distance(x, y):
    return (x - y) ** 2


def calculate_total_value(left, right, x, y, j, kernel_half, distance=l1_distance):
    total = 0
    max_value = 255 ** 2

    for v in range(-kernel_half, kernel_half + 1):
        for u in range(-kernel_half, kernel_half + 1):
            value = max_value
            if (x + u - j) >= 0:
                if distance == l1_distance:
                    value = l1_distance(
                        int(left[y + v, x + u]), int(right[y + v, x + u - j]))
                elif distance == l2_distance:
                    value = l2_distance(
                        int(left[y + v, x + u]), int(right[y + v, x + u - j]))
            total += value

    return total


def window_based_matching_l1(left_img, right_img, disparity_range, kernel_size=5):
    left = cv2.imread(left_img, 0)
    right = cv2.imread(right_img, 0)

    left = left.astype(np.float32)
    right = right.astype(np.float32)

    height, width = left.shape[:2]

    depth = np.zeros((height, width), np.uint8)
    kernel_half = int((kernel_size - 1) / 2)
    scale = 3
    max_value = 255 ** 2

    for y in range(kernel_half, height - kernel_half):
        for x in range(kernel_half, width - kernel_half):
            disparity = 0
            cost_min = max_value

            for j in range(disparity_range):
                total = calculate_total_value(
                    left, right, x, y, j, kernel_half, l1_distance)

                if total < cost_min:
                    cost_min = total
                    disparity = j

            depth[y, x] = disparity * scale

    return depth


def window_based_matching_l2(left_img, right_img, disparity_range, kernel_size=5):
    left = cv2.imread(left_img, 0)
    right = cv2.imread(right_img, 0)

    left = left.astype(np.float32)
    right = right.astype(np.float32)

    height, width = left.shape[:2]

    depth = np.zeros((height, width), np.uint8)
    kernel_half = int((kernel_size - 1) / 2)
    scale = 3
    max_value = 255 ** 2

    for y in range(kernel_half, height - kernel_half):
        for x in range(kernel_half, width - kernel_half):
            disparity = 0
            cost_min = max_value

            for j in range(disparity_range):
                total = calculate_total_value(
                    left, right, x, y, j, kernel_half, l2_distance)

                if total < cost_min:
                    cost_min = total
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

    # Create results directory
    if not os.path.isdir("./results"):
        os.mkdir("./results")
    results_path = r"./results/window_based_matching/"

    '''
    Problem 2: Window-based Matching with Aloe Images
    '''
    left_img_path = r"./data/Aloe_images/Aloe/Aloe_left_1.png"
    right_img_path = r"./data/Aloe_images/Aloe/Aloe_right_1.png"
    disparity_range = 64
    kernel_size = 3

    window_based_result_l1 = window_based_matching_l1(
        left_img_path,
        right_img_path,
        disparity_range,
        kernel_size
    )
    save_results(window_based_result_l1, results_path,
                 "L1", "Aloe1.png", )

    window_based_result_l2 = window_based_matching_l2(
        left_img_path,
        right_img_path,
        disparity_range,
        kernel_size
    )
    save_results(window_based_result_l2, results_path,
                 "L2", "Aloe1.png", )

    '''
    Problem 3: Window-based Matching with challenging Aloe images
    '''
    left_img_path = r"./data/Aloe_images/Aloe/Aloe_left_1.png"
    right_img_path = r"./data/Aloe_images/Aloe/Aloe_right_2.png"
    disparity_range = 64
    kernel_size = 5

    window_based_result_l1 = window_based_matching_l1(
        left_img_path,
        right_img_path,
        disparity_range,
        kernel_size
    )
    save_results(window_based_result_l1, results_path,
                 "L1", "Aloe2.png", )

    window_based_result_l2 = window_based_matching_l2(
        left_img_path,
        right_img_path,
        disparity_range,
        kernel_size
    )
    save_results(window_based_result_l2, results_path,
                 "L2", "Aloe2.png", )

    '''
    NOTE: Độ đo L1 và L2 không có tính chất 'invariant to linear changes'
    (như cosine similarity và correlation coefficient). Do đó, L1 L2 sẽ
    không thể hoạt động tốt với hai ảnh cùng nội dung nhưng có một chút
    khác biệt liên quan đến độ sáng...
    '''
