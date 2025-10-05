import numpy as np
import random
from utils import timing
import json
from numba import jit
from funciones_prueba import drop, ackley, boha1, matya
from funciones_taller import cost_slug_model, cost_genuchten_model

random.seed(42)
np.random.seed(42)


@jit(nopython=False)
def ma(func, m0, bounds, initial_temp=100.0, iter_per_temp=100):
    """
    Clásico Metropolis Algorithm (Simulated Annealing)
    """
    T = initial_temp.copy()
    Em0 = func(m0)
    k = 0

    while T > 0.001:  # criterio de parada
        for _ in range(iter_per_temp):
            m1 = np.zeros_like(m0)
            for j in range(len(m0)):
                m1[j] = np.random.uniform(bounds[j][0], bounds[j][1])
            delta_e = func(m1) - Em0
            # exponent = min(100, max(-100, -delta_e / T))  # clamp para evitar overflow
            P = np.exp(-delta_e / T)

            if delta_e < 0 or random.uniform(0, 1) < P:
                m0 = m1.copy()
                Em0 = func(m0)

        # enfriamiento lineal
        if T > 2:
            T -= 1
        else:
            T -= 0.1

        k += 1

    return m0, Em0, k


@jit(nopython=False)
def vfma(
    func,
    m0,
    bounds,
    initial_temps=np.array([5.0, 5.0]),
    coeficients=np.array([1, 1]),
    iter_per_temp=100,
    NM=2,
):
    """
    Very Fast Metropolis Algorithm
    """
    T = initial_temps.copy()
    C = coeficients
    Em0 = func(m0)
    k = 0
    while np.linalg.norm(T) > 0.001:  # criterio de parada
        for _ in range(iter_per_temp):
            m1 = m0.copy()
            for j in range(NM):
                U = random.uniform(0, 1)
                yi = np.sign(U - 0.5) * T[j] * ((1 + 1 / T[j]) ** (abs(2 * U - 1)) - 1)
                m1[j] = m0[j] + yi * (bounds[j][1] - bounds[j][0])
                m1[j] = min(max(m1[j], bounds[j][0]), bounds[j][1])

            delta_e = func(m1) - Em0
            P = np.exp(-delta_e / T[0])
            if delta_e <= 0 or P > random.uniform(0, 1):
                m0 = m1.copy()
                Em0 = func(m0)

        k += 1
        for j in range(NM):
            T[j] = initial_temps[j] * np.exp(-C[j] * k ** (1 / NM))

    return m0, Em0, k


def get_problem_params(problem: str):
    if problem == "drop":
        bounds = np.array([[-5.0, 5.0], [-5.0, 5.0]])
        initial_temps = np.array([100.0, 100.0])
        num_pairs = 100
        func = drop

    elif problem == "boha1":
        bounds = np.array([[-100, 100], [-100, 100]])
        initial_temps = np.array([100.0, 100.0])
        num_pairs = 100
        func = boha1

    elif problem == "ackley":
        bounds = np.array([[-32, 32], [-32, 32]])
        initial_temps = np.array([100.0, 100.0])
        num_pairs = 100
        func = ackley

    elif problem == "matya":
        bounds = np.array([[-10, 10], [-10, 10]])
        initial_temps = np.array([100.0, 100.0])
        num_pairs = 100
        func = matya

    elif problem == "slug":
        bounds = np.array([[0.001, 0.01], [0.1, 1.0]])
        initial_temps = np.array([100.0, 100.0])
        num_pairs = 100
        func = cost_slug_model

    elif problem == "genuchten":
        bounds = np.array([[0.001, 0.02], [1.0, 10.0]])
        initial_temps = np.array([100.0, 100.0])
        num_pairs = 100
        func = cost_genuchten_model

    else:
        raise ValueError(f"Unknown problem specified: {problem}")

    return bounds, initial_temps, num_pairs, func


@timing
def simulated_annealing(problem, algorithm):

    bounds, initial_temps, num_pairs, func = get_problem_params(problem)

    m0 = [
        np.random.uniform(bounds[0][0], bounds[0][1], num_pairs),
        np.random.uniform(bounds[1][0], bounds[1][1], num_pairs),
    ]
    print("-" * 50)
    print(f"Running {algorithm} for problem: {problem} with {num_pairs} pairs")
    print(f"Bounds: {bounds}")
    results = []
    for i in range(num_pairs):
        m0_pair = np.array([m0[0][i], m0[1][i]])
        if algorithm == "ma":
            m_opt, cost, iters = ma(
                func, m0_pair, bounds, initial_temp=initial_temps[0:1]
            )
        elif algorithm == "vfma":
            m_opt, cost, iters = vfma(
                func, m0_pair, bounds, initial_temps=initial_temps
            )
        print(f"Pair {i}: m = {m_opt}, cost = {cost}, iterations = {iters}")
        results.append((m_opt, cost))

    m1, c = zip(*results)  # Properly unpack the results

    results_list = [
        {"m": m1[i].tolist(), "cost": float(c[i])} for i in range(num_pairs)
    ]

    print("-" * 50)

    # ------------------
    # Save in Json
    # ------------------
    with open(f"results/{problem}_{algorithm}.json", "w") as f:
        json.dump(results_list, f, indent=4)


if __name__ == "__main__":
    problem_list = [
        "drop",
        "ackley",
        "boha1",
        "matya",
        "slug",
        "genuchten",
    ]  # "drop", "ackley", "boha1", "matya", "slug", "genuchten"
    for problem in problem_list:
        simulated_annealing(problem, "vfma")
