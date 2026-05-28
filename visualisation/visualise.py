from metrics.swarm_analysis import load_positions
from swarm_plots import plot_trajectories, plot_error_curve

pos = load_positions("exports/sim0_all_positions.csv")

initial = pos[0]
target = pos[-1]

plot_trajectories(initial, target, pos)
plot_error_curve(pos, target)
