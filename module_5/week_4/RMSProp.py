# Cho hàm số f(w1, w2) = 0.1*w1^2 + 2*w2^2

import numpy as np

def df_w(w):
    """
    Thực hiện tính gradient của dw1 và dw2
    Arguments:
    W -- np.array [w1, w2]
    Returns:
    dW -- np.array [dw1, dw2], array chứa giá trị đạo hàm theo w1 và w2
    """
    #################### YOUR CODE HERE ####################
    dw1 = 0.2 * w[0]
    dw2 = 4 * w[1]

    dW = np.array([dw1, dw2])
    ########################################################

    return dW

def RMSProp(W, dW, lr, S, gamma):
    """
    Thực hiện thuật tóan RMSProp để update w1 và w2
    Arguments:
    W -- np.array: [w1, w2]
    dW -- np.array: [dw1, dw2], array chứa giá trị đạo hàm theo w1 và w2
    lr -- float: learning rate
    S -- np.array: [s1, s2] Exponentially weighted averages bình phương gradients
    gamma -- float: hệ số long-range average
    Returns:
    W -- np.array: [w1, w2] w1 và w2 sau khi đã update
    S -- np.array: [s1, s2] Exponentially weighted averages bình phương gradients sau khi đã cập nhật
    """
    epsilon = 1e-6
    #################### YOUR CODE HERE ####################

    S = gamma * S + (1 - gamma) * dW**2

    W = W - lr * dW / np.sqrt(S + epsilon)
    ########################################################
    return W, S

def train_p1(optimizer, lr, epochs):
    """
    Thực hiện tìm điểm minimum của function (1) dựa vào thuật toán
    được truyền vào từ optimizer
    Arguments:
    optimize : function thực hiện thuật toán optimization cụ thể
    lr -- float: learning rate
    epochs -- int: số lượng lần (epoch) lặp để tìm điểm minimum
    Returns:
    results -- list: list các cặp điểm [w1, w2] sau mỗi epoch (mỗi lần cập nhật)
    """
    # initial
    W = np.array([-5, -2], dtype=np.float32)
    S = np.array([0, 0], dtype=np.float32)
    results = [W]
    #################### YOUR CODE HERE ####################
    # Tạo vòng lặp theo số lần epochs
    # tìm gradient dW gồm dw1 và dw2
    # dùng thuật toán optimization cập nhật w1, w2, s1, s2
    # append cặp [w1, w2] vào list results

    for _ in range(epochs):
        dW = df_w(W)
        W, S = optimizer(W, dW, lr, S, gamma=0.9)
        results.append(W)

    results = np.array(results)
    ########################################################
    return results

results = train_p1(RMSProp, lr=0.3, epochs=30)
print(results)
