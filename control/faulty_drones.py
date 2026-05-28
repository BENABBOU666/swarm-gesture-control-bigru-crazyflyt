import random

def get_faulty_drones(n_drones, fault_level):
    """
    fault_level:
        0 → no fault
        1 → 1% fault
        2 → 2% fault
    """

    if fault_level == 0:
        return []

    # ---------------- FIXED RULES ----------------
    rule_map = {
        7:  {1: 1, 2: 2},
        15: {1: 2, 2: 3},
        25: {1: 3, 2: 5}
    }

    if n_drones not in rule_map:
        # fallback safe percentage (if future swarm size)
        k = int(round((fault_level / 100) * n_drones))
        k = max(1, k)
    else:
        k = rule_map[n_drones][fault_level]

    # ensure validity
    k = min(k, n_drones)

    return random.sample(range(n_drones), k)
