import random
ERROR_MESSAGE = "The length of pred and target must be equal"


def mae_loss(pred, target):
    '''
    This function calculates the Mean Absolute Error (MAE) loss function
    Args:
        pred: list of predicted values
        target: list of target values
    Returns:
        mae: Mean Absolute Error
    '''

    if len(pred) != len(target):
        print(ERROR_MESSAGE)
        return None

    total_loss = 0
    for i in range(len(pred)):
        loss = abs(pred[i] - target[i])
        total_loss += loss
        print(f"loss name: MAE, sample {i}: pred = {
              pred[i]}, target = {target[i]}, loss = {loss}")

    mae = total_loss / len(pred)
    return mae


def mse_loss(pred, target):
    '''
    This function calculates the Mean Squared Error (MSE) loss function
    Args:
        pred: list of predicted values
        target: list of target values
    Returns:
        mse: Mean Squared Error
    '''
    if len(pred) != len(target):
        print(ERROR_MESSAGE)
        return None

    total_loss = 0
    for i in range(len(pred)):
        loss = (pred[i] - target[i]) ** 2
        total_loss += loss
        print(f"loss name: MSE, sample {i}: pred = {
              pred[i]}, target = {target[i]}, loss = {loss}")

    mse = total_loss / len(pred)
    return mse


def rmse_loss(pred, target):
    '''
    This function calculates the Root Mean Squared Error (RMSE) loss function
    Args:
        pred: list of predicted values
        target: list of target values
    Returns:
        rmse: Root Mean Squared Error
    '''
    if len(pred) != len(target):
        print(ERROR_MESSAGE)
        return None

    total_loss = 0
    for i in range(len(pred)):
        loss = (pred[i] - target[i]) ** 2
        total_loss += loss
        print(f"loss name: RMSE, sample {i}: pred = {
              pred[i]}, target = {target[i]}, loss = {loss}")

    rmse = (total_loss / len(pred)) ** 0.5
    return rmse


def calc_loss_function():
    number_samples = input("Input number of samples (integer number): ")
    if number_samples.isnumeric() == False:
        print("number of samples must be an integer")
        return None
    else:
        number_samples = int(number_samples)

    loss_name = input("Loss name (MAE|MSE|RMSE): ")
    if loss_name not in ['MAE', 'MSE', 'RMSE']:
        print(f"{loss_name} is not supported")
        return

    y_pred = [random.uniform(0, 10) for _ in range(number_samples)]
    y_target = [random.uniform(0, 10) for _ in range(number_samples)]

    if loss_name == 'MAE':
        print(f"final MAE: {mae_loss(y_pred, y_target)}")
    elif loss_name == 'MSE':
        print(f"final MSE: {mse_loss(y_pred, y_target)}")
    elif loss_name == 'RMSE':
        print(f"final RMSE: {rmse_loss(y_pred, y_target)}")


if __name__ == "__main__":
    calc_loss_function()
