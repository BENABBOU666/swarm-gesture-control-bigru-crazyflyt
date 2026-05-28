from control.controllers.swarm_simulation import run_simulation

if __name__ == "__main__":

   
    # run_simulation(n_drones=7, noise_mode="baseline", sim_id=1) # baseline : normal conditions
    run_simulation(n_drones=15, noise_mode="formation", sigma=0.5, sim_id=2) # noise 
    # run_simulation(n_drones=25, noise_mode="state", sigma=0.5, sim_id=3) # noise
    # run_simulation(n_drones=25, noise_mode="baseline", fault_mode=True, fault_level=2) # faulty 1%
