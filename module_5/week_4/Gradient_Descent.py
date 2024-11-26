# Cho hàm số f(w1, w2) = 0.1*w1^2 + 2*w2^2

import numpy as np

def df_w(W):
    """
    Thực hiện tính gradient của dw1 và dw2
    Arguments:
    W -- np.array [w1, w2]
    Returns:
    dW -- np.array [dw1, dw2], array chứa giá trị đạo hàm theo w1 và w2
    """
    #################### YOUR CODE HERE ####################
    dw1 = 0.2 * W[0]
    dw2 = 4 * W[1]

    dW = np.array([dw1, dw2])
    ########################################################

    return dW

def sgd(W, dW, lr):
    """
    Thực hiện thuật tóa Gradient Descent để update w1 và w2
    Arguments:
    W -- np.array: [w1, w2]
    dW -- np.array: [dw1, dw2], array chứa giá trị đạo hàm theo w1 và w2
    lr -- float: learning rate
    Returns:
    W -- np.array: [w1, w2] w1 và w2 sau khi đã update
    """
    #################### YOUR CODE HERE ####################
    w1 = W[0] - lr * dW[0]
    w2 = W[1] - lr * dW[1]

    W = np.array([w1, w2])
    ########################################################
    return W

def train_p1(optimizer, lr, epochs):
    """
    Thực hiện tìm điểm minimum của function (1) dựa vào thuật toán
    được truyền vào từ optimizer
    Arguments:
    optimize : function thực hiện thuật toán optimization cụ thể
    lr -- float: learning rate
    epoch -- int: số lượng lần (epoch) lặp để tìm điểm minimum
    Returns:
    results -- list: list các cặp điểm [w1, w2] sau mỗi epoch (mỗi lần cập nhật)
    """

    # initial point
    W = np.array([-5, -2], dtype=np.float32)
    # list of results
    results = [W]

    #################### YOUR CODE HERE ####################
    # Tạo vòng lặp theo số lần epochs
    # tìm gradient dW gồm dw1 và dw2
    # dùng thuật toán optimization cập nhật w1 và w2
    # append cặp [w1, w2] vào list results

    for _ in range(epochs):
        dW = df_w(W)
        W = optimizer(W, dW, lr)
        results.append(W)
    results = np.array(results)

    ########################################################
    return results

result = train_p1(sgd, lr=0.4, epochs=30)
print(result)
