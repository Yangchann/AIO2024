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

def sgd_momentum(W, dW, lr, V, beta):
    """
    Thực hiện thuật tóan Gradient Descent + Momentum để update w1 và w2
    Arguments:
    W -- np.array: [w1, w2]
    dW -- np.array: [dw1, dw2], array chứa giá trị đạo hàm theo w1 và w2
    lr -- float: learning rate
    V -- np.array: [v1, v2] Exponentially weighted averages gradients
    beta -- float: hệ số long-range average
    Returns:
    W -- np.array: [w1, w2] w1 và w2 sau khi đã update
    V -- np.array: [v1, v2] Exponentially weighted averages gradients sau khi đã cập nhật
    """
    #################### YOUR CODE HERE ####################

    V = beta * V + (1 - beta) * dW
    W = W - lr * V
    ########################################################
    return W, V

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
    results = [W]
    #################### YOUR CODE HERE ####################
    # Tạo vòng lặp theo số lần epochs
    # tìm gradient dW gồm dw1 và dw2
    # dùng thuật toán optimization cập nhật w1, w2, v1, v2
    # append cặp [w1, w2] vào list results

    for _ in range(epochs):
        dW = df_w(W)
        W, V = optimizer(W, dW, lr, V, 0.5)
        results.append(W)

    results = np.array(results)
    ########################################################
    return results

result  = train_p1(sgd_momentum, lr=0.6, epochs=30)
print(result)
