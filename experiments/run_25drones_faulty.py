from control.controllers.formation_controller_faulty_25drones import run_simulation

if __name__ == "__main__":

    #run_simulation(n_drones=25, noise_mode="baseline", fault_mode=False, sim_id=0) # normal
    #run_simulation(n_drones=25, noise_mode="formation", sigma=0.3, sim_id=1) # noise
    run_simulation(n_drones=25, noise_mode="baseline", fault_mode=True, fault_level=2) # faulty 1%
    #run_simulation(n_drones=25, noise_mode="baseline", fault_mode=True, fault_level=2, sim_id=3) # faulty 2%
