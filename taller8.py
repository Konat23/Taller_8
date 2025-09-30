import numpy as np
import random
from concurrent.futures import ThreadPoolExecutor
import json

from funciones_prueba import drop, ackley, boha1, matya
from funciones_taller import cost_slug_model, cost_genuchten_model

random.seed(42)
np.random.seed(42)


def vfma(
    func,
    m0,
    bounds,
    i,
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
                m1[j] = np.clip(m1[j], bounds[j][0], bounds[j][1])

            delta_e = func(m1) - Em0
            P = np.exp(-delta_e / T[0])
            if delta_e <= 0 or P > random.uniform(0, 1):
                m0 = m1.copy()
                Em0 = func(m0)

        k += 1
        for j in range(NM):
            T[j] = initial_temps[j] * np.exp(-C[j] * k ** (1 / NM))

    print(f"Pair {i}: m = {m0}, cost = {Em0}, iterations = {k}")
    return m0, Em0


def simulated_annealing(problem, n):
    if problem == "drop":
        bounds = [[-5, 5], [-5, 5]]
        num_pairs = 100
        func = drop

    elif problem == "boha1":
        bounds = [[-100, 100], [-100, 100]]
        num_pairs = 100
        func = boha1

    elif problem == "ackley":
        bounds = [[-32, 32], [-32, 32]]
        num_pairs = 100
        func = ackley

    elif problem == "matya":
        bounds = [[-10, 10], [-10, 10]]
        num_pairs = 100
        func = matya

    elif problem == "slug":
        bounds = [[0.001, 0.01], [0.1, 1.0]]
        num_pairs = 50
        func = cost_slug_model

    elif problem == "genuchten":
        bounds = [[0.001, 0.02], [1, 10]]
        num_pairs = 50
        func = cost_genuchten_model

    else:
        raise ValueError(f"Unknown problem specified: {problem}")

    m0 = [
        np.random.uniform(bounds[0][0], bounds[0][1], num_pairs),
        np.random.uniform(bounds[1][0], bounds[1][1], num_pairs),
    ]

    results = []
    for i in range(num_pairs):
        m0_pair = [m0[0][i], m0[1][i]]
        result = vfma(func, m0_pair, bounds, i)
        results.append(result)

    m1, c = zip(*results)  # Properly unpack the results

    if problem in ["slug", "genuchten"]:
        results_list = [{"m": m1[i], "cost": c[i][0]} for i in range(num_pairs)]
    else:
        results_list = [{"m": m1[i], "cost": c[i]} for i in range(num_pairs)]
    # ------------------
    # Save in Json
    # ------------------
    with open(f"results_{problem}_{n}.json", "w") as f:
        json.dump(results_list, f, indent=4)


if __name__ == "__main__":
    # problem = "drop"  # problems: "boha1","matya","ackley", "drop",, slug, genuchten

    problem_list = ["drop"]
    for problem in problem_list:
        simulated_annealing(problem, "vfma")
