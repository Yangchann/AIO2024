import math


def cal_factorial(n):
    '''
    This function calculates the factorial of a number
    Args:
        n: integer number
    Returns:
        factorial: factorial of n
    '''
    if n == 0:
        return 1
    else:
        return n * cal_factorial(n-1)


def approx_sin(x, n):
    sin = 0
    for i in range(n):
        sin += (-1) ** i * x ** (2*i+1) / cal_factorial(2*i+1)
    return sin


def approx_cos(x, n):
    cos = 0
    for i in range(n):
        cos += (-1) ** i * x ** (2*i) / cal_factorial(2*i)
    return cos


def approx_sinh(x, n):
    sinh = 0
    for i in range(n):
        sinh += x ** (2 * i + 1) / cal_factorial(2 * i + 1)
    return sinh


def approx_cosh(x, n):
    cosh = 0
    for i in range(n):
        cosh += x ** (2 * i) / cal_factorial(2 * i)
    return cosh


if __name__ == "__main__":
    x = float(input("Enter x (radian): "))
    n = int(input("Enter n (iterations): "))
    if n <= 0:
        print("n must be greater than zero")
    else:
        print(f"sin({x})={math.sin(x)}, approx_sin({
              x}, {n})={approx_sin(x, n)}")
        print(f"cos({x})={math.cos(x)}, approx_cos({
              x}, {n})={approx_cos(x, n)}")
        print(f"sinh({x})={math.sinh(x)}, approx_sinh({
              x}, {n})={approx_sinh(x, n)}")
        print(f"cosh({x})={math.cosh(x)}, approx_cosh({
              x}, {n})={approx_cosh(x, n)}")
