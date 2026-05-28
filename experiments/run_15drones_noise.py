from control.controllers.formation_controller_noise_15drones import run_simulation

if __name__ == "__main__":

   
    #run_simulation(n_drones=7, noise_mode="baseline", sim_id=1) # normal conditions
    run_simulation(n_drones=15, noise_mode="formation", sigma=0.5, sim_id=2) # noise
    #run_simulation(n_drones=25, noise_mode="state", sigma=0.5, sim_id=3) # noise 
