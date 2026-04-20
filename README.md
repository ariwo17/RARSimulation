# Estimating the Gradient Noise Scale in Distributed SGD via Linear Sketches

This repository contains the experimental framework for my Final Year Project (MEng Mathematical Computation) at University College London (UCL). 

It features a custom **Ring-Allreduce Distributed Data Parallel (DDP) Simulator** engineered to explore the intersection of Gradient Noise Scale (GNS) tracking, critical batch size ($B_{\mathrm{crit}}$) analysis, and communication-efficient gradient compression.

---

## Core Contributions

This repository was forked from [https://github.com/QErywan/RARSimulation.git](https://github.com/QErywan/RARSimulation). The original code has been heavily refactored to serve as a highly automated, analytical ML testbed. Key engineering novelties include:

1. **Zero-Overhead GNS Estimator:** Implements the McCandlish et al. two-batch method to track gradient variance and signal. Crucially, the estimator is adapted to compute norms directly on **Count-Sketched** gradients, bypassing the massive memory overhead of decompression.
2. **Adaptive-k Sparsification Scheduler:** Dynamically adjusts Random-k gradient sparsity during training using real-time GNS estimates as a proxy for SGD noise tolerance.
3. **Automated Search Utilities:** A suite of CLI wrappers (e.g., `grid_runner.py`, `localsgd_runner.py`) designed for extracting time-to-accuracy plots, Pareto frontiers and optimal learning rates across hundreds of sequential runs.
4. **Algorithmic Fixes:** Corrected flaws in previous iterations of the simulator, including global weight de-synchronisation, in-place gradient vanishing bugs, and stateful optimiser drift.

## Table of Contents

- [Core Contributions](#core-contributions)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)

---

## Prerequisites

- **Python** >= 3.9
- `pip` (usually bundled with Python)

---

## Installation

1. Clone the repo  
   ```bash
   git clone https://github.com/ariwo17/RARSimulation.git
   cd RARSimulation

2. Create a Virtual Environment
    ```bash
    python3 -m venv .venv

3. Activate the Virtual Environment
    ```bash
    source .venv/bin/activate

4. Upgrade pip
    ```bash
    pip install --upgrade pip

5. Install project requirements
    ```bash
    pip install -r requirements.txt


## Usage

All simulation parameters are managed via a Command Line Interface (CLI) using `argparse` within `ringallreduce_sim.py`.

### Simulation Parameters

| Parameter             | Description                                                                                                   |
|-----------------------|---------------------------------------------------------------------------------------------------------------|
| `seed`                | Random seed for reproducibility.                                                                              |
| `gpu`                 | GPU ID to use (0 for first GPU, -1 for CPU).                                                                  |
| `num_rounds`          | Maximum number of communication rounds.                                                                       |
| `num_clients`         | Number of participating clients in Ring-Allreduce toplogy.                                                    |
| `target_acc`          | Target SMA training accuracy for early stopping.                                                              |
| `test_every`          | Frequency (in rounds) to evaluate test accuracy.                                                              |
| `optim`               | Optimiser algorithm (sgd, momentum, adam). Set to sgd by default.                                             |
| `client_train_steps`  | Number of local training steps per client before communication.                                               |
| `client_batch_size`   | Batch size used by each client (for both training and testing).                                               |
| `lr`                  | Learning rate used during training.                                                                           |
| `lr_type`             | Type of learning rate scheduling. Options: `'const'`, `'step_decay'`, `'exp_decay'`, `'acc_decay'`.           |
| `milestones`          | Comma-separated accuracy milestones for accuracy-activated step decay (acc_decay).                            |
| `net`                 | CNN model architecture to use, e.g. `'ResNet9'`.                                                              |
| `dataset`             | Dataset to use, e.g. `'CIFAR10'`.                                                                             |
| `nbits`               | Bits per coordinate used in theoretical bandwidth tracking.                                                   |
| `compression_scheme`  | Compression/decompression strategy. Options: `'none'`, `'vector_topk'`, '`randomk`', `'chunk_topk_recompress'`, `'chunk_topk_single'`, `'csh'`, `'cshtopk_actual'`, `'cshtopk_estimate'`. |
| `sketch_col`          | Sketch width / number of buckets (Count Sketch).                                                              |
| `sketch_row`          | Number of independent hash rows (Set to 1 for Feature Hashing).                                               |
| `k`                   | Number of elements to retain for `'randomk'` or `'vector_topk.'`).                                            |
| `error_feedback`      | Toggle for error feedback mechanism (currently not implemented).                                              |
| `adaptive_k`          | Flag to enable GNS-guided dynamic sparsity. Set to False by default.                                          |
| `alpha`               | Noise tolerance threshold (ratio of compression var to SGD var). Set to 0.01 by default.                      |
| `adaptive_milestones` | Training accuracy milestones where K is re-evaluated. Set to [0.0,60.0,90.0] by default.                      |
| `data_per_client`     | Data distribution method across clients. Options: `'sequential'`, `'label_per_client'`, `'iid'`.              |
| `folder`              | Target directory for saving .pt result logs                                                                   |


### Running the Simulation

1. To run the simulation:
    ```bash
    python3 ringallreduce_sim.py

Examples:
1. Run a synchronous DDP simulation on CIFAR-10 using ResNet9 and Count Sketch ($\times 10$ compression):
    ```python3 ringallreduce_sim.py --dataset CIFAR10 --net ResNet9 --optim momentum --lr 0.075 --compression_scheme csh --sketch_col 490324 --num_clients 8
2. Run a simulation where Random-K sparsity is dynamically adjusted based on the GNS estimate when specific training accuracy milestones are hit:
    ```python3 ringallreduce_sim.py --dataset CIFAR10 --net ResNet9 --compression_scheme randomk --adaptive_k --alpha 0.001 --adaptive_milestones 0.0,50.0,80.0

