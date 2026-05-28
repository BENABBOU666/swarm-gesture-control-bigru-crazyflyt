import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


# =========================================================
# LOAD FROM swarm_metrics
# =========================================================
def load_positions(csv_path):

    df = pd.read_csv(csv_path)

    if "t" not in df.columns:
        raise ValueError("Expected column 't' from swarm_metrics export")

    positions = []

    for _, group in df.groupby("t"):
        group = group.sort_values("drone")
        positions.append(group[['x', 'y', 'z']].values)

    return np.array(positions)

# =========================================================
# ERROR FUNCTION
# =========================================================
def compute_error(current, target):
    current = np.array(current)
    target = np.array(target)

    return np.mean(np.linalg.norm(current - target, axis=1))

# =========================================================
# PROCESS SINGLE SIMULATION
# =========================================================
def process_simulation(csv_path, target_positions, sim_index):

    positions = load_positions(csv_path)

    if len(positions) == 0:
        raise ValueError(f"No data found in {csv_path}")

    errors = []

    for pos in positions:
        err = compute_error(pos, target_positions)
        errors.append(err)

    cumulative_error = np.mean(errors)

    # =====================================================
    # SAVE RESULTS
    # =====================================================
    os.makedirs("results", exist_ok=True)

    df_errors = pd.DataFrame({
        "Iteration": np.arange(len(errors)),
        "Error": errors
    })

    df_errors.to_csv(
        f"results/per_iteration_errors_sim{sim_index+1}.csv",
        index=False
    )

    # =====================================================
    # PLOT
    # =====================================================
    os.makedirs("plots", exist_ok=True)

    plt.figure(figsize=(7, 4))
    plt.plot(errors)
    plt.title(f"Simulation {sim_index+1} - Error Curve")
    plt.xlabel("Iteration")
    plt.ylabel("Error")
    plt.grid(True)

    plt.savefig(f"plots/sim{sim_index+1}_error.png")
    plt.close()

    print(f"[OK] Sim {sim_index+1} → Error = {cumulative_error:.6f}")

    return cumulative_error, errors


# =========================================================
# MULTI SIMULATION ANALYSIS
# =========================================================
def main(sim_files, target_positions):

    if len(sim_files) == 0:
        print("No simulation files provided.")
        return

    all_errors = []
    all_iterations = []
    summary = []

    for i, file in enumerate(sim_files):

        err, iters = process_simulation(file, target_positions, i)

        all_errors.append(err)
        all_iterations.append(iters)

        summary.append({
            "Simulation": f"Sim {i+1}",
            "Cumulative Error": err
        })

    # =====================================================
    # AVERAGE ERROR
    # =====================================================
    avg_error = np.mean(all_errors)

    summary.append({
        "Simulation": "Average",
        "Cumulative Error": avg_error
    })

    os.makedirs("results", exist_ok=True)

    pd.DataFrame(summary).to_csv("results/summary.csv", index=False)

    # =====================================================
    # COMBINED ITERATION PLOT
    # =====================================================
    plt.figure(figsize=(8, 4))

    for i, iters in enumerate(all_iterations):
        plt.plot(iters, label=f"Sim {i+1}")

    plt.title("All Simulations - Error Comparison")
    plt.xlabel("Iteration")
    plt.ylabel("Error")
    plt.grid(True)
    plt.legend()

    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/all_simulations_error.png")
    plt.close()

    print("\n========== FINAL RESULTS ==========")
    print(f"Average Error: {avg_error:.6f}")
    print("Results saved in /results and /plots")
