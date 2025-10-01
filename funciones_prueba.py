import numpy as np
from numba import jit


@jit(nopython=True)
def drop(xx):
    """INPUT:
    xx = [x1, x2]
    xi e [-5.12, 5.12]
    f(x*)= -1 at x*=(0,0)
    """
    x1, x2 = xx[0], xx[1]

    frac1 = 1 + np.cos(12 * np.sqrt(x1**2 + x2**2))
    frac2 = 0.5 * (x1**2 + x2**2) + 2

    return -frac1 / frac2


@jit(nopython=True)
def ackley(xx, a=20, b=0.2, c=2 * np.pi):
    """Ackley function.
    INPUT:
    xx: array of input values
    a, b, c: optional parameters with default values
    """
    d = len(xx)

    # Calcular sum1 sin usar generadores
    sum1 = 0.0
    for i in range(d):
        sum1 += xx[i] * xx[i]

    # Calcular sum2 sin usar generadores
    sum2 = 0.0
    for i in range(d):
        sum2 += np.cos(c * xx[i])

    term1 = -a * np.exp(-b * np.sqrt(sum1 / d))
    term2 = -np.exp(sum2 / d)

    return term1 + term2 + a + np.exp(1.0)


@jit(nopython=True)
def boha1(xx):
    """Boha1 function.
    INPUT:
    xx: list of input values [x1, x2]
    """
    x1, x2 = xx[0], xx[1]

    term1 = x1**2
    term2 = 2 * x2**2
    term3 = -0.3 * np.cos(3 * np.pi * x1)
    term4 = -0.4 * np.cos(4 * np.pi * x2)

    return term1 + term2 + term3 + term4 + 0.7


@jit(nopython=True)
def matya(xx):
    """Matya function.
    INPUT:
    xx: list of input values [x1, x2]
    """
    x1, x2 = xx[0], xx[1]

    term1 = 0.26 * (x1**2 + x2**2)
    term2 = -0.48 * x1 * x2

    return term1 + term2


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    X, Y = np.meshgrid(np.linspace(-5.12, 5.12, 100), np.linspace(-5.12, 5.12, 100))
    Z = drop([X, Y])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="viridis")
    plt.show()
