import numpy as np
from metrics.swarm_analysis import main
from formation import FORMATION_MATRIX_7

sim_files = [
    "exports/sim0_all_positions.csv"
]

target = np.array(FORMATION_MATRIX_7[0])  # FIXED TARGET

main(sim_files, target)
