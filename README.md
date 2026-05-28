# Gesture-Based Swarm Formation/Flight Control Using CrazyFLyt Simulator


## Overview

This repository contains the implementation, simulation framework, trained perception model, and evaluation tools associated with a manuscript currently under review:


The framework integrates:

* Deep-learning-based hand gesture recognition,
* Real-time perception and communication via ZeroMQ,
* Swarm formation and flight control using artificial potential fields,
* Bidirectional repulsive–attractive multi-drones coordination,
* Robustness evaluation under baseline, uncertainty and fault conditions.

The system is implemented using the CrazyFlyt simulator and supports evaluation across:

* 7 drones,
* 15 drones,
* 25 drones.

The repository is designed to ensure reproducibility, modular experimentation, and independent verification of results.

---

# Repository Structure

```text
project_root/
│
├── training/
│   ├── bigru.h5
│   ├── training_bigru.py
│   ├── bigru_model.png
│   └── dataset_RNN/
│
├── perception/
│   └── capture_recognise.py
│
├── control/
│   ├── formation.py
│   ├── faulty_drones.py
│   └── controllers/
│       ├── swarm_simulation.py
│       ├── flight_controller.py
│       ├── formation_controller_normal_7drones.py
│       ├── formation_controller_noise_15drones.py
│       └── formation_controller_faulty_25drones.py
│   
├──experiments/
│      ├── run_simulation.py
│      ├── run_flight_simulation.py
│      ├── run_7drones_normal.py
│      ├── run_15drones_noise.py
│      └── run_25drones_faulty.py
│
├──metrics/
│      ├── swarm_metrics.py
│      └── swarm_analysis.py
│      
├──visualisation/
│      ├── swarm_plots.py
│      ├── analysis.py
│      └── visualise.py
│
│
├── simulation_videos/
│   └── simulation_videos/
│
├── requirements.txt
├── LICENSE
├── CITATION.cff
└── README.md
```

---

# System Architecture 

### System-Level Overview of the Proposed Gesture-Based Drone Swarm Interaction


  <img src="figures/SystemLevelOverview_HSI.PNG)" width="700"/>



**Figure .** System-level overview of the proposed gesture-based drone swarm interaction framework.

---

# 1. Training Module

Path:

```text
training/
```

This module implements the gesture recognition pipeline based on a Bidirectional Gated Recurrent Unit (BiGRU) architecture: 

##  Key Components
* training_bigru.py: training and evaluation pipeline,
* best_model_bigru.h5: pre-trained model used in all experiments,
* dataset_RNN/: sequential gesture dataset.

---

# 2. Perception Module

Path: 
```text
perception/capture_recognise.py
```

This module corresponds to the gesture acquisition and recognition layers of the system architecture. 

Implements real-time gesture recognition:

* MediaPipe-based hand landmark extraction,
* BiGRU inference,
* Real-time classification,
* Communication of recognised gestures via ZeroMQ (ZMQ).

---

# 3. Dependency Requirement: CrazyFlyt Simulator

This project requires the CrazyFlyt simulator.

The simulator is not included in this repository.

---

# 3.1. Installation

Clone the official repository:
```bash
git clone https://github.com/jjshoots/CrazyFlyt.git`
cd CrazyFlyt
pip install -r requirements.txt
```

# 3.2. Integration

Copy this repository into:

```text
CrazyFlyt/examples/
```

Example structure:

```text
CrazyFlyt/
├── examples/
│   ├── control/
│   ├── training/
│   ├── perception/
│   └── ...
```
All experiments must be executed within the CrazyFlie simulation environment.

---

# 4. Control Module

Path:

```text
control/
```

This module implements tools for swarm coordination, formation control, and evaluation. It corresponds to the swarm formation and flight control layer in the system architecture.

---

# 4.1 Formation Definition

```text
control/formation.py
```

Defines:

* Formation matrices for 7, 15, and 25 drones,
* Mapping from gestures to formation indices: acts as the decision interface between perception and swarm control.

   Recognised gestures are mapped into swarm-level commands using a predefined mapping:

   * Formation Mode: gestures define target swarm geometric formations
   * Flight Mode: gestures define direct motion commands

   This mapping ensures a structured translation from human intent to swarm behaviour.
   
Core structures:

```python
GESTURE_TO_FORMATION
FORMATION_MATRIX_7
FORMATION_MATRIX_15
FORMATION_MATRIX_25
```

---

# 4.2 Swarm Simulation Engine

```text
control/controller/swarm_simulation.py
```

Implements the main swarm controller using artificial potential fields:

* Attractive forces for formation convergence,
* Repulsive forces for collision avoidance,
* Real-time control loop,
* ZeroMQ communication interface,
* Multi-scenario robustness evaluation.

## Experimental Modes

| Mode        | Description                                   |
| ----------- | --------------------------------------------- |
| `baseline`  | Ideal conditions (no perturbation) perturbation         |
| `formation` | Noise applied to target formation coordinates |
| `state`     | Noise applied to drone dynamics updates          |



# 4.3 Experimental Setups

* run_7drones.py: baseline evaluation,
* run_15drones_noise.py: robustness under perception noise,
* run_25drones_faulty.py: fault-tolerant swarm evaluation.
* run_simulation.py: full swarm formation evaluation.
* run_flight_simulation.py: swarm flight control evaluation.

This module corresponds to the validation through simulation using CrazyFlyt layers of the system architecture.
---

# 4.4 Metrics Module

```text
metrics/swarm_metrics.py
```

Provides quantitative evaluation:

* trajectory length,
* convergence error,
* energy consumption,
* inter-agent distances,
* execution time,
* Global swarm performance indicators.

Outputs are stored in CSV format for reproducibility.

---

# 4.5 Analysis Module

```text
metrics/swarm_analysis.py
```

Performs post-processing:

* convergence analysis,
* statistical aggregation,
* multi-run comparison,
* automated report generation.

---

# 4.6 Visualisation Module

```text
visualisation/swarm_plots.py
```

Generates:

* trajectory visualisations,
* convergence curves,
* energy plots,
* Comparative performance graphs.

---

# 4.7 Simulation Visualisation

```text
visualisation/visualise.py
```

Provides real-time and post-simulation swarm rendering.

---

# 5. Simulation Examples

Path:

```text
simulation_videos/
```

Contains demonstration videos illustrating:

Includes qualitative demonstrations of:

* formation transitions,
* flight, 
* convergence behaviour,
* robustness under noise,
* fault scenarios.

These examples are included to support the qualitative assessment of the proposed framework.

---

# Experimental Methodology

The framework evaluates:

1. Formation convergence accuracy,
2. Collision avoidance capability,
3. Robustness to uncertainty,
4. Fault tolerance,
5. Scalability across swarm sizes.


# Swarm Configurations

Three swarm sizes are considered:

| Swarm Size | Scenario                   |
| ---------- | -------------------------- |
| 7 drones   | Baseline formation control |
| 15 drones  | Noise robustness           |
| 25 drones  | Fault tolerance            |

---

# Fault Injection Protocol

Faulty drone experiments simulate inactive agents.

```text
control/fualty_drone.py
```

Fault levels are defined as:

| Swarm Size | Fault Level 1   | Fault Level 2   |
| ---------- | --------------- | --------------- |
| 7 drones   | 1 faulty drone  | 2 faulty drones |
| 15 drones  | 2 faulty drones | 3 faulty drones |
| 25 drones  | 3 faulty drones | 5 faulty drones |

Faults simulate:

* actuator failures,
* communication loss,
* localisation errors,
* energy depletion.

---

# Dependencies

The framework was developed using Python 3.8+.

Main dependencies are:

```text
numpy
pandas
matplotlib
tensorflow
keras
opencv-python
mediapipe
pyzmq
```

---

# Running Experiments

## a. Perception

```bash
python perception/capture_recognise.py
```

---

## b. Run Baseline Simulation (7drones)

```bash
python experiments/run_7drones_normal.py
```

---

## c. Run Noise Robustness Simulation (15 drones)

```bash
python experiments/run_15drones_noise.py
```

---

## d. Run Fault-Tolerance Simulation (25 drones)

```bash
python experiments/run_25drones_faulty.py
```

---

# Reproducibility

Random seeds are fixed:

```bash
np.random.seed(42)
```

---

# Notes

This repository is provided solely for:

* scientific validation,
* reproducibility verification,
* academic review purposes.

The implementation prioritises:

* readability,
* modularity,
* experimental transparency,
* reproducibility.


---

# Citation

If this repository contributes to your research, please cite:


```bibtex
@software{benabbou2026,
  author       = {BENABBOU, F.B.E. , BENSLIMANE, S.M. , KHALDI, B.},
  title        = {Gesture-Based Swarm Formation/Flight Control Using CrazyFLyt Simulator},
  year         = {2026},
  version      = {1.0},
  publisher    = {GitHub},
  url          = {https://github.com/BENABBOU666/swarm-gesture-control-bigru-crazyflyt}
}
```

---

# Author

**BENABBOU F.B.E , BENSLIMANE S , KHALDI B**
---

