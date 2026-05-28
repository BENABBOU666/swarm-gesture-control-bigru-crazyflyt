"""
Flight Control Module

Pipeline:
1. Select random formation index (1–18)
2. Apply flight commands on top of that formation
3. Evaluate baseline, noise and fault robustness

BENABBOU F.B.E
"""

import os
import zmq
import numpy as np
import random
from signal import SIGINT, signal
from CrazyFlyt import Simulator
from control.formation import (
    FORMATION_MATRIX_7,
    FORMATION_MATRIX_15,
    FORMATION_MATRIX_25
)
from control.faulty_drones import get_faulty_drones

# =========================================================
# SAFETY
# =========================================================
def shutdown_handler(*_):
    print("CTRL-C → safe shutdown")
    os._exit(0)

signal(SIGINT, shutdown_handler)


# =========================================================
# FLIGHT MODEL
# =========================================================
def fly(positions, distance=1.0, direction="z"):
    new_positions = positions.copy()

    if direction == "x":
        new_positions[:, 0] += distance
    elif direction == "y":
        new_positions[:, 1] += distance
    elif direction == "z":
        new_positions[:, 2] += distance
    else:
        raise ValueError("Invalid direction")

    return new_positions


# =========================================================
# COMMAND PARSER
# =========================================================
def parse_direction(message):
    m = message.lower()

    if m == "up":
        return 1.0, "z"
    if m == "down":
        return -1.0, "z"
    if m == "left":
        return 1.0, "y"
    if m == "right":
        return -1.0, "y"
    if m == "go":
        return 1.0, "x"
    if m == "back":
        return -1.0, "x"

    return 0.0, "z"



# =========================================================
# FORMATION SELECTOR
# =========================================================
def select_random_formation():
    """
    Index meaning:
    0  -> takeoff
    19 -> stop
    1-18 -> valid formations
    """
    return random.randint(1, 18)


def get_matrix(n):
    return (
        FORMATION_MATRIX_7 if n == 7 else
        FORMATION_MATRIX_15 if n == 15 else
        FORMATION_MATRIX_25
    )
# =========================================================
# MAIN
# =========================================================
def run_simulation(n_drones=7, noise_mode="baseline",fault_mode=False, fault_level=0, sigma=0.1, sim_id=0):
    indx = 0
    FORMATION_MATRIX = get_matrix(n_drones)

    # =====================================================
    # INIT FORMATION
    # =====================================================
    idx = select_random_formation()
    base_formation = np.array(FORMATION_MATRIX[idx])

    positions = base_formation.copy()

    swarm = Simulator(
        start_states=np.hstack((positions, np.zeros((n_drones, 1))))
    )

    swarm.set_pos_control(True)
    swarm.arm([True] * n_drones)
    swarm.set_setpoints(np.hstack((positions, np.zeros((n_drones, 1)))))
    swarm.sleep(0.25)

    # =====================================================
    # FAULT SETUP
    # =====================================================
    faulty = get_faulty_drones(n_drones, fault_level)
    print(f"[FAULTY] {faulty}")

    # =====================================================
    # ZMQ
    # =====================================================
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5555")

    step_fraction = 0.3
    num_iterations = 70
    epsilon = 1e-3

    print(f"[RUN] drones={n_drones} | noise={noise_mode} | fault={fault_level}")

    # =====================================================
    # LOOP
    # =====================================================
    while True:

        msg = socket.recv().decode().strip()
        socket.send_string("OK")

        if msg.lower() == "stop":
            break

        # =================================================
        # FLIGHT CONTROL LAYER
        # =================================================
        if msg.lower() == "takeoff":
            positions[:, 2] = 0.5

        elif msg.lower() == "land":
            positions[:, 2] = 0.0

        else:
            direction_map = {
                "up": (1.0, "z"),
                "down": (-1.0, "z"),
                "left": (1.0, "y"),
                "right": (-1.0, "y"),
                "go": (1.0, "x"),
                "back": (-1.0, "x")
            }

            if msg in direction_map:
                d, direction = direction_map[msg]

                if noise_mode == "formation_noise":
                    d += np.random.normal(0, sigma)

                if direction == "x":
                    positions[:, 0] += d
                elif direction == "y":
                    positions[:, 1] += d
                elif direction == "z":
                    positions[:, 2] += d

        # =================================================
        # STATE NOISE 
        # =================================================
        if noise_mode == "state_noise":
            positions += np.random.normal(0, sigma, positions.shape)

        # =================================================
        # FAULT LAYER
        # =================================================
        for d in faulty:
            positions[d] = positions[d]  # frozen drone

        # =================================================
        # SEND TO SIMULATOR
        # =================================================
        setpoints = np.hstack((positions, np.zeros((n_drones, 1))))

        swarm.set_setpoints(setpoints)
        swarm.sleep(2)

    swarm.arm([False] * n_drones)
    print(f"[DONE] Simulation {sim_id}")
