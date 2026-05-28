import numpy as np
import matplotlib.pyplot as plt
import os


# =========================================================
# 3D TRAJECTORIES
# =========================================================
def plot_trajectories(initial, target, all_positions, save_path=None):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    all_positions = np.array(all_positions)

    n_drones = initial.shape[0]

    for i in range(n_drones):
        ax.plot(
            all_positions[:, i, 0],
            all_positions[:, i, 1],
            all_positions[:, i, 2]
        )

    ax.scatter(initial[:, 0], initial[:, 1], initial[:, 2],
               c='red', label="Initial")

    ax.scatter(target[:, 0], target[:, 1], target[:, 2],
               c='blue', label="Target")

    ax.set_title("Swarm Trajectories")
    ax.legend()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)

    plt.show()


# =========================================================
# CONVERGENCE ERROR (FROM POSITIONS)
# =========================================================
def plot_error_curve(all_positions, target, save_path=None):

    all_positions = np.array(all_positions)
    target = np.array(target)

    errors = []

    for pos in all_positions:
        err = np.linalg.norm(pos - target, axis=1)
        errors.append(np.max(err))

    plt.figure()
    plt.plot(errors)
    plt.title("Convergence Error (Max per iteration)")
    plt.xlabel("Iteration")
    plt.ylabel("Max Error")
    plt.grid()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)

    plt.show()


# =========================================================
# ENERGY CURVE
# =========================================================
def plot_energy(energy_list, save_path=None):

    plt.figure()
    plt.plot(energy_list)
    plt.title("Energy Evolution")
    plt.xlabel("Iteration")
    plt.ylabel("Energy")
    plt.grid()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)

    plt.show()


# =========================================================
# GLOBAL METRICS BAR PLOT
# =========================================================
def plot_global_metrics(metrics_dict, save_path=None):

    keys = list(metrics_dict.keys())
    values = list(metrics_dict.values())

    plt.figure()
    plt.bar(keys, values)
    plt.xticks(rotation=45)
    plt.title("Global Swarm Metrics")
    plt.grid(axis='y')

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)

    plt.show()
