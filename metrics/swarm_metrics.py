import numpy as np
import pandas as pd
import os

EXPORT_DIR = "exports"
os.makedirs(EXPORT_DIR, exist_ok=True)


# =========================================================
# INTERNAL SAVE FUNCTION
# =========================================================
def _save(df, name, mode="w"):
    path = os.path.join(EXPORT_DIR, name)

    if mode == "a" and os.path.exists(path):
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)

    print(f"[Saved] {path}")


# =========================================================
# POSITIONS
# =========================================================
def save_positions(positions, sim_id=0, name="final_positions.csv"):
    positions = np.array(positions)

    df = pd.DataFrame(positions, columns=["x", "y", "z"])
    _save(df, f"sim{sim_id}_{name}")


def save_all_positions(all_positions, sim_id=0):
    rows = []

    for t, pos in enumerate(all_positions):
        pos = np.array(pos)
        for i, p in enumerate(pos):
            rows.append([t, i, p[0], p[1], p[2]])

    df = pd.DataFrame(rows, columns=["t", "drone", "x", "y", "z"])
    _save(df, f"sim{sim_id}_all_positions.csv")


# =========================================================
# ITERATION TIME
# =========================================================
def save_iteration_times(times, sim_id=0):
    df = pd.DataFrame({
        "iteration": range(len(times)),
        "time_sec": times
    })
    _save(df, f"sim{sim_id}_iteration_times.csv")


# =========================================================
# FORCES
# =========================================================
def save_forces(forces, sim_id=0, name="forces.csv"):
    rows = []

    for t, f in enumerate(forces):
        f = np.array(f)
        for i, v in enumerate(f):
            rows.append([t, i, v[0], v[1], v[2]])

    df = pd.DataFrame(rows, columns=["t", "drone", "fx", "fy", "fz"])
    _save(df, f"sim{sim_id}_{name}")


# =========================================================
# TRAJECTORY METRICS
# =========================================================
def compute_trajectory_lengths(all_pos):
    all_pos = np.array(all_pos)

    n = all_pos.shape[1]
    lengths = np.zeros(n)

    for t in range(len(all_pos) - 1):
        lengths += np.linalg.norm(all_pos[t + 1] - all_pos[t], axis=1)

    return lengths


def save_trajectory_lengths(lengths, sim_id=0):
    df = pd.DataFrame({
        "drone": range(len(lengths)),
        "trajectory_length": lengths
    })
    _save(df, f"sim{sim_id}_trajectory_lengths.csv")


# =========================================================
# DISTANCES
# =========================================================
def compute_distances(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.linalg.norm(a - b, axis=1)


def save_distances(d, sim_id=0, name="distances.csv"):
    df = pd.DataFrame({
        "drone": range(len(d)),
        "distance": d
    })
    _save(df, f"sim{sim_id}_{name}")


# =========================================================
# ACCURACY
# =========================================================
def compute_accuracy(traj, optimal):
    traj = np.array(traj)
    optimal = np.array(optimal)

    return np.array([
        1 - (t / o) if o > 1e-6 else 0
        for t, o in zip(traj, optimal)
    ])


def save_accuracy(acc, sim_id=0):
    df = pd.DataFrame({
        "drone": range(len(acc)),
        "accuracy": acc
    })
    _save(df, f"sim{sim_id}_trajectory_accuracy.csv")


# =========================================================
# GLOBAL METRICS 
# =========================================================
def compute_global(target, final, rep_events, times, energy, distance):
    target = np.array(target)
    final = np.array(final)

    error = np.linalg.norm(target - final, axis=1)

    return {
        "mean_error": float(np.mean(error)),
        "max_error": float(np.max(error)),
        "stability": float(np.var(error)),
        "collision_events": float(rep_events),
        "collision_rate": float(rep_events),
        "energy_efficiency": float(distance / energy) if energy > 1e-9 else 0.0,
        "total_time": float(np.sum(times)),
        "final_error_norm": float(np.linalg.norm(error))
    }


def save_global(metrics, sim_id=0):
    df = pd.DataFrame([metrics])
    _save(df, f"sim{sim_id}_global_metrics.csv")
