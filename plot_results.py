import json
import matplotlib.pyplot as plt
import os


def plot_results(
    problem,
    PATH,
    n,
    title="",
    bounds=[[0.001, 0.01], [0.1, 1.0]],
    ground_truth=[0.00207, 0.585],
):
    try:
        with open(PATH, "r") as f:
            results = json.load(f)

        m_values = [result["m"] for result in results]
        costs = [result["cost"] for result in results]

        # Extract m1 and m2 values for plotting
        m1_values = [m[0] for m in m_values]
        m2_values = [m[1] for m in m_values]
        if problem in ["slug", "genuchten"]:
            plt.figure(figsize=(14, 6))
        else:
            plt.figure(figsize=(7, 6))

        # Subplot 1: Scatter plot with bounds
        plt.subplot(1, 1 if problem not in ["slug", "genuchten"] else 2, 1)
        plt.scatter(m1_values, m2_values, c="blue", label="Results")
        plt.scatter(
            ground_truth[0],
            ground_truth[1],
            c="red",
            marker="x",
            s=100,
            label="Ground Truth",
        )

        mean_m1 = sum(m1_values) / len(m1_values)
        mean_m2 = sum(m2_values) / len(m2_values)
        plt.scatter(
            mean_m1,
            mean_m2,
            c="orange",
            marker="o",
            s=100,
            label="Mean of Results",
        )

        plt.axvline(bounds[0][0], color="green", linestyle="--", label="Bounds")
        plt.axvline(bounds[0][1], color="green", linestyle="--")
        plt.axhline(bounds[1][0], color="green", linestyle="--")
        plt.axhline(bounds[1][1], color="green", linestyle="--")
        plt.xlabel("m1")
        plt.ylabel("m2")
        plt.title(f"Results for {problem} (with bounds)")
        plt.legend()
        plt.grid()

        if problem in ["slug", "genuchten"]:
            # Subplot 2: Scatter plot without bounds
            plt.subplot(1, 2, 2)
            plt.scatter(m1_values, m2_values, c="blue", label="Results")
            for m1, m2, cost in zip(m1_values, m2_values, costs):
                plt.text(m1, m2, f"{cost:.2f}", fontsize=8, ha="right")
            plt.scatter(
                ground_truth[0],
                ground_truth[1],
                c="red",
                marker="x",
                s=100,
                label="Ground Truth",
            )
            plt.scatter(
                mean_m1,
                mean_m2,
                c="orange",
                marker="o",
                s=100,
                label="Mean of Results",
            )
            plt.xlabel("m1")
            plt.ylabel("m2")
            plt.title(f"Results for {problem} (zoomed in)")
            plt.legend()
            plt.grid()

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Save the plot as a PDF in the "plots" directory
        os.makedirs("plots", exist_ok=True)
        plt.savefig(f"plots/results_{problem}_{n}.pdf")
        plt.show()

    except FileNotFoundError:
        print(f"Error: results_{problem}.json not found.")
    except json.JSONDecodeError:
        print("Error: Failed to decode JSON file.")


def compute_metrics(problem, PATH):
    try:
        with open(PATH, "r") as f:
            results = json.load(f)

        costs = [result["cost"] for result in results]
        best_cost = min(costs)
        worst_cost = max(costs)
        avg_cost = sum(costs) / len(costs)
        print(f"Metrics for {problem}:")
        print(f"Best Cost: {best_cost}")
        print(f"Worst Cost: {worst_cost}")
        print(f"Average Cost: {avg_cost}")

        # Mean and standard deviation of m1 and m2
        m_values = [result["m"] for result in results]
        m1_values = [m[0] for m in m_values]
        m2_values = [m[1] for m in m_values]
        mean_m1 = sum(m1_values) / len(m1_values)
        mean_m2 = sum(m2_values) / len(m2_values)
        std_m1 = (sum((x - mean_m1) ** 2 for x in m1_values) / len(m1_values)) ** 0.5
        std_m2 = (sum((x - mean_m2) ** 2 for x in m2_values) / len(m2_values)) ** 0.5
        print(f"Mean m1: {mean_m1}, Std Dev m1: {std_m1}")
        print(f"Mean m2: {mean_m2}, Std Dev m2: {std_m2}")
        return mean_m1, std_m1, mean_m2, std_m2
    except FileNotFoundError:
        print(f"Error: results_{problem}.json not found.")
    except json.JSONDecodeError:
        print("Error: Failed to decode JSON file.")


def get_problem_bounds(problem):
    if problem == "drop":
        bounds = [[-5, 5], [-5, 5]]
        ground_truth = [0, 0]
    elif problem == "boha1":
        bounds = [[-100, 100], [-100, 100]]
        ground_truth = [0, 0]
    elif problem == "ackley":
        bounds = [[-32, 32], [-32, 32]]
        ground_truth = [0, 0]
    elif problem == "matya":
        bounds = [[-10, 10], [-10, 10]]
        ground_truth = [0, 0]
    elif problem == "slug":
        bounds = [[0.001, 0.01], [0.1, 1.0]]
        ground_truth = [0.00207, 0.585]
    elif problem == "genuchten":
        bounds = [[0.001, 0.02], [1, 10]]
        ground_truth = [0.012605, 1.853943]
    else:
        raise ValueError(f"Unknown problem specified: {problem}")
    return bounds, ground_truth


def run_problem(algorithm, problem):
    PATH = f"results/{problem}_{algorithm}.json"
    mean_m1, std_m1, mean_m2, std_m2 = compute_metrics(problem, PATH)
    # Save txt with metrics
    with open(f"metrics/{problem}_{algorithm}.txt", "w") as f:
        f.write(f"Metrics for {problem} problem:\n")
        f.write(f"Mean m1: {mean_m1}, Std Dev m1: {std_m1}\n")
        f.write(f"Mean m2: {mean_m2}, Std Dev m2: {std_m2}\n")

    bounds, ground_truth = get_problem_bounds(problem)

    # Update the plot_results call to use the dynamically set bounds and ground_truth
    # plot_results(problem, PATH, algorithm, bounds=bounds, ground_truth=ground_truth)
    plot_comparison(problem, bounds, ground_truth)


def plot_comparison(problem, bounds, ground_truth):
    try:
        # Load results for both algorithms
        PATH_ma = f"results/{problem}_ma.json"
        PATH_vfma = f"results/{problem}_vfma.json"

        with open(PATH_ma, "r") as f:
            results_ma = json.load(f)

        with open(PATH_vfma, "r") as f:
            results_vfma = json.load(f)

        # Extract m1 and m2 values for both algorithms
        m_values_ma = [result["m"] for result in results_ma]
        m1_values_ma = [m[0] for m in m_values_ma]
        m2_values_ma = [m[1] for m in m_values_ma]

        m_values_vfma = [result["m"] for result in results_vfma]
        m1_values_vfma = [m[0] for m in m_values_vfma]
        m2_values_vfma = [m[1] for m in m_values_vfma]

        # Create scatter plot
        plt.figure(figsize=(8, 6))
        plt.scatter(m1_values_ma, m2_values_ma, c="blue", label="MA Results")
        plt.scatter(m1_values_vfma, m2_values_vfma, c="green", label="VFMA Results")
        plt.scatter(
            ground_truth[0],
            ground_truth[1],
            c="red",
            marker="x",
            s=100,
            label="Ground Truth",
        )

        plt.axvline(bounds[0][0], color="gray", linestyle="--", label="Bounds")
        plt.axvline(bounds[0][1], color="gray", linestyle="--")
        plt.axhline(bounds[1][0], color="gray", linestyle="--")
        plt.axhline(bounds[1][1], color="gray", linestyle="--")

        plt.xlabel("m1")
        plt.ylabel("m2")
        plt.title(f"Comparison of MA and VFMA Results for {problem}")
        plt.legend()
        plt.grid()

        # Save the plot as a PDF in the "plots" directory
        os.makedirs("plots", exist_ok=True)
        plt.savefig(f"plots/comparison_{problem}.pdf")
        plt.show()

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except json.JSONDecodeError:
        print("Error: Failed to decode JSON file.")


def comparation(problem):
    bounds, ground_truth = get_problem_bounds(problem)
    plot_comparison(problem, bounds, ground_truth)


if __name__ == "__main__":
    os.makedirs("plots", exist_ok=True)
    os.makedirs("metrics", exist_ok=True)
    problem_list = [
        "drop",
        "ackley",
        "boha1",
        "matya",
        "slug",
        "genuchten",
    ]  # "drop", "ackley", "boha1", "matya", "slug", "genuchten"
    for problem in problem_list:
        comparation(problem)
