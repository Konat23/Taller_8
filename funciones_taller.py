import numpy as np
from numba import jit


@jit(nopython=True)
def cost_slug_model(xx):
    S = xx[0]  # S
    T = xx[1]  # T
    Q = 50.0
    d = 60.0

    # Crear arrays usando funciones soportadas por numba
    t = np.array([5.0, 10.0, 20.0, 30.0, 40.0, 50.0]).reshape(-1, 1)
    h = np.array([0.72, 0.49, 0.30, 0.20, 0.16, 0.12]).reshape(-1, 1)

    # Calcular head_eq usando operaciones elemento a elemento
    head_eq = (Q / (4.0 * np.pi * T * t)) * np.exp((-(d**2) * S) / (4.0 * T * t))

    # Calcular la norma usando operaciones básicas
    diff = head_eq - h
    r = np.sqrt(np.sum(diff * diff))
    return r


data = np.load("vgdata.npz")
h = data["h"]
theta = data["theta"]


@jit(nopython=True)
def cost_genuchten_model(xx):
    alpha = xx[0]
    n = xx[1]
    theta_s = 0.44
    theta_r = 0.09

    theta_eq = theta_r + (theta_s - theta_r) / (1 + (-alpha * h) ** n) ** (1 - 1 / n)

    diff = theta_eq - theta
    r = np.sqrt(np.sum(diff * diff))
    return r


if __name__ == "__main__":
    ##
    # xx = np.array([0.00207, 0.585])
    # r = cost_slug_model(xx)
    # print("Result:", r)
    # xx = np.array([0.1, 10])
    # xx = np.array([[0.1, 1.1, 0.00207], [10, 10.1, 0.585]])

    ##
    xx = np.array([0.012605, 1.853943])
    r = cost_genuchten_model(xx)
    print("Result:", r)
