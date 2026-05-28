"""
Formation Control Module for Swarm of N drones
3 Noise Experimental Cases:
- baseline
- formation_noise
- state_noise

BENABBOU F.B.E
"""

import os
import time
import zmq
import numpy as np
from signal import SIGINT, signal
from CrazyFlyt import Simulator

from control.formation import (
    FORMATION_MATRIX_7,
    FORMATION_MATRIX_15,
    FORMATION_MATRIX_25,
    GESTURE_TO_FORMATION
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
def map_gesture(message: str) -> int:
    return GESTURE_TO_FORMATION.get(message.strip(), 0)


# =========================================================
# REPULSIVE FORCE
# =========================================================
def apply_repulsive(positions, r_rep=0.25, k_rep=0.5, eta_rep=0.9):

    forces = np.zeros_like(positions)
    n = len(positions)

    for i in range(n):
        for j in range(i + 1, n):

            diff = positions[i] - positions[j]
            dist = np.linalg.norm(diff)

            if 1e-6 < dist < r_rep:

                direction = diff / dist
                magnitude = k_rep * (1.0 / dist - 1.0 / r_rep) * (1.0 / dist**2)

                f = magnitude * direction

                forces[i] += f
                forces[j] -= f

    return eta_rep * forces


# =========================================================
# MAIN SIMULATION
# =========================================================
def run_simulation(
    n_drones=15,
    noise_mode="baseline", 
    sigma=0.5,
    sim_id=0
):

    # =====================================================
    # SELECT FORMATION MATRIX
    # =====================================================
    if n_drones == 7:
        FORMATION_MATRIX = FORMATION_MATRIX_7
    elif n_drones == 15:
        FORMATION_MATRIX = FORMATION_MATRIX_15
    elif n_drones == 25:
        FORMATION_MATRIX = FORMATION_MATRIX_25
    else:
        raise ValueError("Unsupported n_drones (use 7, 15, or 25)")

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

    # =====================================================
    # ZMQ
    # =====================================================
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5555")

    step_fraction = 0.3
    num_iterations = 70
    epsilon = 1e-3

    print(f"Controller ready | drones={n_drones} | mode={noise_mode}")

    # =====================================================
    # MAIN LOOP
    # =====================================================
    while True:

        message = socket.recv().decode().strip()
        socket.send_string("OK")

        if message.lower() == "stop":
            break

        gesture_id = map_gesture(message)

        # base formation
        base_target = np.array(FORMATION_MATRIX[gesture_id])

        # safety check (VERY IMPORTANT for reviewers)
        if base_target.shape[0] != n_drones:
            raise ValueError(
                f"Mismatch: formation has {base_target.shape[0]} points but n_drones={n_drones}"
            )

        # =====================================================
        # NOISE MODES
        # =====================================================
        if noise_mode == "baseline":
            #ideal system (no uncertainty)
            target_positions = base_target

        elif noise_mode == "formation":
            # perception / communication noise
            target_positions = base_target + np.random.normal(0, sigma, base_target.shape)

        elif noise_mode == "state":
            # state uncertainty (apply noise to robot positions instead) # add at level of position update
            target_positions = base_target 

        else:
            raise ValueError("Invalid noise_mode")



        # reset
        positions = initial_positions.copy()

        # =====================================================
        # FORMATION LOOP
        # =====================================================
        for step in range(num_iterations):

            rep = apply_repulsive(positions)
            att = step_fraction * (target_positions - positions)

            new_positions = positions + rep + att

            # state noise (only for actuator uncertainty)
            if noise_mode == "state":
                new_positions += np.random.normal(0, sigma, new_positions.shape)

            positions = new_positions

            swarm.reshuffle(
                np.hstack((positions, np.zeros((n_drones, 1))))
            )
            swarm.sleep(0.3)

            error = np.linalg.norm(positions - target_positions, axis=1)

            if np.max(error) < epsilon:
                print(f"Converged in {step} iterations")
                break

    swarm.arm([False] * n_drones)
    print("Swarm shutdown complete")
