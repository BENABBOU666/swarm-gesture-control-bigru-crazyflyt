"""
Unified Swarm Formation Simulation Engine 
Supports:
- baseline
- formation_noise (perception)
- state_noise (actuator)
- faulty drones (1% / 2%)

Fully integrated metrics export for scientific evaluation.

BENABBOU F.B.E
"""

import os
import time
import zmq
import numpy as np
import random
from signal import SIGINT, signal
from CrazyFlyt import Simulator
from control.faulty_drones import get_faulty_drones

from control.formation import (
    FORMATION_MATRIX_7,
    FORMATION_MATRIX_15,
    FORMATION_MATRIX_25,
    GESTURE_TO_FORMATION
)

from metrics.swarm_metrics import (
    save_iteration_times,
    save_all_positions,
    save_forces,
    compute_trajectory_lengths,
    save_trajectory_lengths,
    compute_distances,
    save_distances,
    compute_global,
    save_global
)

# =========================================================
# SAFETY
# =========================================================
def shutdown_handler(*_):
    print("CTRL-C detected → shutting down safely")
    os._exit(0)

signal(SIGINT, shutdown_handler)


# =========================================================
# GESTURE MAPPING
# =========================================================
def map_gesture(msg):
    return GESTURE_TO_FORMATION.get(msg.strip(), 0)


# =========================================================
# REPULSIVE FORCE MODEL
# =========================================================
def apply_repulsive(positions, r_rep=0.25, k_rep=0.5, eta=0.9):

    forces = np.zeros_like(positions)
    n = len(positions)

    for i in range(n):
        for j in range(i + 1, n):

            diff = positions[i] - positions[j]
            dist = np.linalg.norm(diff)

            if 1e-6 < dist < r_rep:
                direction = diff / dist
                mag = k_rep * (1/dist - 1/r_rep) * (1/dist**2)
                f = mag * direction

                forces[i] += f
                forces[j] -= f

    return eta * forces


# =========================================================
# MAIN SIMULATION
# =========================================================
def run_simulation(
    n_drones=7,
    noise_mode="baseline",   # baseline | formation_noise | state_noise
    fault_level=0,           # 0 | 1 | 2
    sigma=0.3,
    sim_id=0
):

    # ---------------- FORMATION MATRIX ----------------
    FORMATION_MATRIX = (
        FORMATION_MATRIX_7 if n_drones == 7 else
        FORMATION_MATRIX_15 if n_drones == 15 else
        FORMATION_MATRIX_25
    )

    np.random.seed(42)

    initial_positions = np.random.uniform(-2, 2, (n_drones, 3))
    initial_positions[:, 2] = 0

    swarm = Simulator(
        start_states=np.hstack((initial_positions, np.zeros((n_drones, 1))))
    )

    swarm.set_pos_control(True)
    swarm.arm([True] * n_drones)

    swarm.set_setpoints(np.hstack((initial_positions, np.zeros((n_drones, 1)))))
    swarm.sleep(0.25)

    # ---------------- ZMQ ----------------
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5555")

    step_fraction = 0.3
    num_iterations = 70
    epsilon = 1e-3

    print(f"[RUN] drones={n_drones} | noise={noise_mode} | fault={fault_level}")

    # =========================================================
    # MAIN LOOP
    # =========================================================
    while True:

        msg = socket.recv().decode().strip()
        socket.send_string("OK")

        if msg.lower() == "stop":
            break

        gesture_id = map_gesture(msg)
        base_target = np.array(FORMATION_MATRIX[gesture_id])

        # ---------------- NOISE ----------------
        if noise_mode == "baseline":
            target = base_target

        elif noise_mode == "formation_noise":
            target = base_target + np.random.normal(0, sigma, base_target.shape)

        elif noise_mode == "state_noise":
            target = base_target

        else:
            raise ValueError("Invalid noise_mode")

        # ---------------- FAULT MODEL ----------------
        faulty = get_faulty_drones(n_drones, fault_level)
        print(f"[FAULTY DRONES] {faulty}")

        # RESET
        positions = initial_positions.copy()

        # LOGS
        all_positions = []
        iteration_times = []
        repulsive_log = []
        attractive_log = []

        total_energy = 0
        total_distance = 0

        # =========================================================
        # FORMATION LOOP
        # =========================================================
        for step in range(num_iterations):

            start = time.time()

            rep = apply_repulsive(positions)
            att = step_fraction * (target - positions)

            new_pos = positions + rep + att

            # ---------------- STATE NOISE ----------------
            if noise_mode == "state_noise":
                new_pos += np.random.normal(0, sigma, new_pos.shape)

            # ---------------- FAULT INJECTION ----------------
            for d in faulty:
                new_pos[d] = positions[d]  # frozen drone

            # LOGGING
            all_positions.append(new_pos.copy())
            repulsive_log.append(rep)
            attractive_log.append(att)

            total_distance += np.linalg.norm(new_pos - positions, axis=1).sum()
            total_energy += np.linalg.norm(att, axis=1).sum()

            positions = new_pos

            swarm.reshuffle(np.hstack((positions, np.zeros((n_drones, 1)))))
            swarm.sleep(0.3)

            iteration_times.append(time.time() - start)

            if np.max(np.linalg.norm(positions - target, axis=1)) < epsilon:
                print(f"Converged in {step} iterations")
                break

    # =========================================================
    # EXPORT METRICS
    # =========================================================
    save_all_positions(all_positions, sim_id)
    save_iteration_times(iteration_times, sim_id)

    final = positions

    save_distances(compute_distances(final, target), sim_id)
    save_trajectory_lengths(compute_trajectory_lengths(all_positions), sim_id)

    metrics = compute_global(
        target,
        final,
        rep_events=0,
        times=iteration_times,
        energy=total_energy,
        distance=total_distance
    )

    save_global(metrics, sim_id)

    save_forces(repulsive_log, sim_id, "repulsive.csv")
    save_forces(attractive_log, sim_id, "attractive.csv")

    print(f"[DONE] Simulation {sim_id} exported")

    swarm.arm([False] * n_drones)
    print("Swarm shutdown complete")
