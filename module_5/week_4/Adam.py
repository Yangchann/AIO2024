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

def Adam(W, dW, lr, V, S, beta1, beta2, t):
    """
    Thực hiện thuật tóan Adam để update w1 và w2
    Arguments:
    W -- np.array: [w1, w2]
    dW -- np.array: [dw1, dw2], array chứa giá trị đạo hàm theo w1 và w2
    lr -- float: learning rate
    V -- np.array: [v1, v2] Exponentially weighted averages gradients
    S -- np.array: [s1, s2] Exponentially weighted averages bình phương gradients
    beta1 -- float: hệ số long-range average cho V
    beta2 -- float: hệ số long-range average cho S
    t -- int: lần thứ t update (bắt đầu bằng 1)
    Returns:
    W -- np.array: [w1, w2] w1 và w2 sau khi đã update
    V -- np.array: [v1, v2] Exponentially weighted averages gradients sau khi đã cập nhật
    S -- np.array: [s1, s2] Exponentially weighted averages bình phương gradients sau khi đã cập nhật
    """
    epsilon = 1e-6
    #################### YOUR CODE HERE ####################
    V = beta1 * V + (1 - beta1) * dW
    S = beta2 * S + (1 - beta2) * dW**2

    V_corr = V / (1 - beta1**t)
    S_corr = S / (1 - beta2**t)

    W = W - lr * V_corr / (np.sqrt(S_corr) + epsilon)
    ########################################################
    return W, V, S

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
    V = np.array([0, 0], dtype=np.float32)
    S = np.array([0, 0], dtype=np.float32)
    results = [W]
    #################### YOUR CODE HERE ####################
    # Tạo vòng lặp theo số lần epochs
    # tìm gradient dW gồm dw1 và dw2
    # dùng thuật toán optimization cập nhật w1, w2, s1, s2, v1, v2
    # append cặp [w1, w2] vào list results
    # các bạn lưu ý mỗi lần lặp nhớ lấy t (lần thứ t lặp) và t bất đầu bằng 1

    for t in range(1, epochs + 1):
        dW = df_w(W)
        W, V, S = optimizer(W, dW, lr, V, S, 0.9, 0.999, t)
        results.append(W)

    results = np.array(results)
    ########################################################
    return results

results = train_p1(Adam, lr=0.2, epochs=30)
print(results)
