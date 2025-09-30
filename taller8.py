import numpy as np
import random
from concurrent.futures import ThreadPoolExecutor
import json

from funciones_prueba import drop, ackley, boha1, matya
from funciones_taller import cost_slug_model, cost_genuchten_model

random.seed(42)
np.random.seed(42)


def vfma(func, m0, bounds, i, initial_temp=np.array([100, 100]), iter_per_temp=100):
    """
    Very Fast metropolis algorithm
    """
    T = initial_temp
    Em0 = func(m0)
    m1 = m0.copy()

    while T > 0:
        for _ in range(iter_per_temp):
            m1 = [
                np.clip(
                    m0[0] + np.random.uniform(-bounds[0][1] / 10, bounds[0][1] / 10),
                    bounds[0][0],
                    bounds[0][1],
                ),
                np.clip(
                    m0[1] + np.random.uniform(-bounds[1][1] / 10, bounds[1][1] / 10),
                    bounds[1][0],
                    bounds[1][1],
                ),
            ]
            delta_e = func(m1) - Em0
            # Clamp the value of -delta_e / T to avoid overflow
            exponent = min(100, max(-100, -delta_e / T))
            P = np.exp(exponent)

            if delta_e < 0 or random.uniform(0, 1) < P:
                m0 = m1
                Em0 = func(m0)
        if T >= 2:
            T -= 1
        else:
            T -= 0.1
    print(f"Pair {i}: m = {m0}, cost = {Em0}")
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

    print(f"Heristic parameters for {problem}:")
    for i in range(num_pairs):
        print(f"m = {m1[i]}, cost = {c[i]}")
    if problem in ["slug", "genuchten"]:
        results_list = [{"m": m1[i], "cost": c[i][0]} for i in range(num_pairs)]
    else:
        results_list = [{"m": m1[i], "cost": c[i]} for i in range(num_pairs)]
    with open(f"results_{problem}{n}.json", "w") as f:
        json.dump(results_list, f, indent=4)


if __name__ == "__main__":
    # problem = "drop"  # problems: drop, boha1, ackley, matya, slug, genuchten

    problem_list = ["genuchten"]
    for problem in problem_list:
        simulated_annealing(problem, 1)
        simulated_annealing(problem, 2)
