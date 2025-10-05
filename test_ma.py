# experimentos.py
import numpy as np
import time
import json
from taller8 import ma, get_problem_params

np.random.seed(42)


# Configuraciones a probar
initial_temps_list = [
    np.array([50.0]),
    np.array([5.0]),
    np.array([3.0]),
]


N_REP = 50  # número de repeticiones por experimento


def run_experiments():
    bounds, initial_temps_default, num_pairs, func = get_problem_params("genuchten")

    results_list = []  # Lista para acumular los resultados

    for init_temps in initial_temps_list:
        print("=" * 70)
        print(f"Experiment MA | init_temps={init_temps}")

        all_costs = []
        times = []
        all_optimal_values = []  # Lista para almacenar los valores óptimos
        m0 = [
            np.random.uniform(bounds[0][0], bounds[0][1], N_REP),
            np.random.uniform(bounds[1][0], bounds[1][1], N_REP),
        ]

        for rep in range(N_REP):
            m0_pair = np.array([m0[0][rep], m0[1][rep]])
            start = time.time()
            m_opt, cost, iters = ma(func, m0_pair, bounds, initial_temp=init_temps)
            elapsed = time.time() - start
            all_costs.append(cost)
            times.append(elapsed)
            all_optimal_values.append(m_opt.tolist())  # Guardar el valor óptimo

            print(
                f"[Rep {rep+1:02d}/{N_REP}] cost={cost:.6f}, "
                f"time={elapsed:.4f}s, iters={iters}"
            )

        avg_time = np.mean(times)
        avg_optimal = np.mean(
            all_optimal_values, axis=0
        )  # Promedio de los valores óptimos

        print("-" * 70)
        print(f"Resultados finales para init_temps={init_temps}")
        print(f"Tiempo promedio: {avg_time:.4f} s")
        print(f"Promedio de valores óptimos: {avg_optimal}")

        # Acumular resultados en la lista
        results = {
            "init_temps": init_temps.tolist(),
            "avg_time": avg_time,
            "avg_optimal": avg_optimal.tolist(),
        }
        results_list.append(results)

        print("=" * 70 + "\n")

    # Guardar todos los resultados en un archivo JSON al final
    with open("test_ma_results.json", "w") as f:
        json.dump(results_list, f, indent=2)


if __name__ == "__main__":
    run_experiments()
