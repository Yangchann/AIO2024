import math


def is_number(n):
    '''
    This function checks if the input is a number
    Args:
        n: input value
    Returns:
        True if n is a number, False otherwise
    '''
    try:
        float(n)
    except ValueError:
        return False
    return True


def calc_sig(x):
    return 1 / (1 + math.exp(-x))


def calc_relu(x):
    return max(0, x)


def calc_elu(x, alpha=0.01):
    return x if x > 0 else alpha*(math.exp(x) - 1)


def calc_activation_function():
    '''
    This function calculates the value of the activation function
    Args:
        x: input value
        activation function name: name of the activation function
    Returns:
        value of the activation function
    Constraints:
        x must be a number, activation function name must be 'sigmoid', 'relu', 'elu'
    '''
    x = input("Input x = ")
    if not is_number(x):
        print("x must be a number")
        return

    function_name = input("Input activation function (sigmoid|relu|elu): ")
    if function_name not in ['sigmoid', 'relu', 'elu']:
        print(f"{function_name} is not supported")
        return

    if function_name == 'sigmoid':
        result = calc_sig(float(x))
    elif function_name == 'relu':
        result = calc_relu(float(x))
    elif function_name == 'elu':
        result = calc_elu(float(x))

    print(f"{function_name} f({x}) = {result}")
    return result


if __name__ == "__main__":
    calc_activation_function()
