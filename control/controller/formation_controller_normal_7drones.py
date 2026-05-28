"""
Formation Control Module for CrazyFlyt Swarm Simulator
(Bidirectional repulsive + attractive forces)

BENABBOU F.B.E
"""

import os
import time
import zmq
import numpy as np
from CrazyFlyt import Simulator

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
# CORE FUNCTION
# =========================================================
def run_simulation(n_drones=15, noise_level=0.0, sim_id=0, zmq_port=5555):

    # ---------------- FORMATION ----------------
    FORMATION_MATRIX = (
        FORMATION_MATRIX_7 if n_drones == 7 else
        FORMATION_MATRIX_15 if n_drones == 15 else
        FORMATION_MATRIX_25 if n_drones == 25 else
        (_ for _ in ()).throw(ValueError("Unsupported swarm size"))
    )

    np.random.seed(42)

    initial_positions = np.random.uniform(-2, 2, (n_drones, 3))
    initial_positions[:, 2] = 0

    start_states = np.hstack((initial_positions, np.zeros((n_drones, 1))))

    swarm = Simulator(start_states=start_states)
    swarm.set_pos_control(True)
    swarm.arm([True] * n_drones)
    swarm.set_setpoints(start_states)
    swarm.sleep(0.25)

    # ---------------- ZMQ ----------------
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://*:{zmq_port}")

    # ---------------- PARAMETERS ----------------
    step_fraction = 0.3
    num_iterations = 70
    epsilon = 1e-3

    print(f"[Sim {sim_id}] Controller ready...")

    # =====================================================
    # MAIN LOOP
    # =====================================================
    while True:

        message = socket.recv().decode().strip()
        socket.send_string("OK")

        if message.lower() == "stop":
            break

        gesture_id = GESTURE_TO_FORMATION.get(message, 0)
        target_positions = np.array(FORMATION_MATRIX[gesture_id])

        positions = initial_positions.copy()

        all_positions = []
        iteration_times = []
        repulsive_log = []
        attractive_log = []

        total_energy = 0
        total_distance = 0

        for step in range(num_iterations):

            start = time.time()

            rep = apply_repulsive(positions)
            att = step_fraction * (target_positions - positions)

            # optional noise injection 
            if noise_level > 0:
                att += np.random.normal(0, noise_level, att.shape)

            new_positions = positions + rep + att

            all_positions.append(new_positions.copy())
            repulsive_log.append(rep)
            attractive_log.append(att)

            total_distance += np.linalg.norm(new_positions - positions, axis=1).sum()
            total_energy += np.linalg.norm(att, axis=1).sum()

            positions = new_positions

            swarm.reshuffle(np.hstack((positions, np.zeros((n_drones, 1)))))
            swarm.sleep(0.3)

            iteration_times.append(time.time() - start)

            if np.max(np.linalg.norm(positions - target_positions, axis=1)) < epsilon:
                print(f"[Sim {sim_id}] Converged at step {step}")
                break

    # =====================================================
    # EXPORT METRICS
    # =====================================================

    save_all_positions(all_positions, sim_id)
    save_iteration_times(iteration_times, sim_id)

    final = positions

    save_distances(
        compute_distances(final, target_positions),
        sim_id
    )

    save_trajectory_lengths(
        compute_trajectory_lengths(all_positions),
        sim_id
    )

    save_global(
        compute_global(
            target_positions,
            final,
            rep_events=0,
            times=iteration_times,
            energy=total_energy,
            distance=total_distance
        ),
        sim_id
    )

    save_forces(repulsive_log, sim_id, "repulsive.csv")
    save_forces(attractive_log, sim_id, "attractive.csv")

    swarm.arm([False] * n_drones)
    print(f"[Sim {sim_id}] Shutdown complete")


# =========================================================
# LOCAL TEST (OPTIONAL)
# =========================================================
#if __name__ == "__main__":
    #run_simulation(n_drones=15, noise_level=0.1, sim_id=1)
