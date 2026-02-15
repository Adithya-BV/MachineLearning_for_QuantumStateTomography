# Machine Learning for Quantum State Tomography (QST)

**Author:** Bollu Venkata Adithya  
**Date:** Winter 2025

## 1. Project Overview

### Problem Statement
Quantum State Tomography (QST) is the process of reconstructing the quantum state (density matrix $\rho$) of a system from a series of measurements. As the number of qubits ($n$) increases, the dimension of the Hilbert space grows exponentially ($2^n \times 2^n$), making standard tomography methods computationally expensive and prone to statistical errors.

This project investigates scalable and physically constrained methods for QST, moving from traditional linear inversion to modern machine learning approaches.

### Objectives
1.  **Understand QST Fundamentals**: Implement basic Linear Inversion tomography on single-qubit systems (Assignment 1).
2.  **Develop Neural Network Models**: Build a physics-informed neural network (Cholesky-MLP) to predict valid density matrices (Assignment 2).
3.  **Analyze Scalability**: Benchmark the performance and runtime of tomography pipelines as system size grows, addressing the "Curse of Dimensionality" (Assignment 3).

---

## 2. Methodology & Workflow

The project consists of three main stages, consolidated from Assignments 1-3.

### Stage 1: Classical Baselines & Data Generation
-   **Measurement Theory**: adopted **Pauli Projective Measurements** (Z, X, Y bases) over SIC-POVMs for hardware compatibility.
-   **Data Generation**: created datasets of random quantum states (pure and mixed) and simulated measurement shots (multinomial distribution) to mimic real quantum hardware noise.
-   **Linear Inversion**: Implemented Maximum Likelihood Estimation (MLE) based linear reconstruction as a baseline:
    $$ \hat{\rho} = \frac{1}{2} \left( I + \sum_{i \in \{x,y,z\}} \langle \sigma_i \rangle \sigma_i \right) $$

### Stage 2: Neural Network QST (Cholesky-MLP)
-   **Architecture**: A Multilayer Perceptron (MLP) takes measurement probabilities as input and outputs a latent representation of the density matrix.
-   **Physical Constraints**: To ensure the predicted state $\rho$ is Positive Semi-Definite (PSD) and trace-one, the network predicts a lower triangular Cholesky factor $L$:
    $$ \rho = \frac{L L^\dagger}{\text{Tr}(L L^\dagger)} $$
    This guarantees physical validity by design, unlike unconstrained linear inversion which can produce non-physical states (negative eigenvalues) due to noise.

### Stage 3: Scalability & Software Engineering
-   **Pipeline Design**: Modularized code into reproducible scripts (`src/`) and notebooks.
-   **Serialization**: Implemented robust model saving/loading mechanisms using `pickle` to handle intermediate states.
-   **Benchmarking**: High-resolution timing analysis of state reconstruction across varying qubit counts ($n=2$ to $n=12$), demonstrating the exponential cost of full state simulation.

---

## 3. Directory Structure

```text
Open_Project_Winter_2025/
|-- data/           # Generated datasets (Pauli measurements, density matrices)
|   |-- .npy, .npz formats
|
|-- models/         # Trained Neural Network Checkpoints
|   |-- best_model.pt
|
|-- notebooks/      # Interactive Analysis & Tutorials
|   |-- Assignment_1.ipynb      # Basics & Linear Inversion
|   |-- Assignment_3.ipynb      # Scalability & Ablation Studies
|
|-- src/            # Modular Source Code
|   |-- data_gen.py             # Quantum Dataset Generation
|   |-- model.py                # Cholesky-MLP Architecture
|   |-- train.py                # Training Loop
|   |-- validate_benchmarks.py  # Validation utility
|
|-- results/        # Generated Plots & Simulation Artifacts
|   |-- hardware_simulation.vcd
|
|-- FINAL_REPORT.md # Detailed Technical Report & Results
|-- AI_USAGE.md     # AI Tools Disclosure
|-- README.md       # This file
```

## 4. Key Results Summary
*See `FINAL_REPORT.md` for detailed LaTeX tables and plots.*

-   **High Fidelity**: The Cholesky-MLP achieved **>97.9%** fidelity on standard benchmark states (|0⟩, |1⟩, |+⟩, etc.).
-   **Physical Validity**: 100% of generated states satisfied physical constraints (Hermitian, Trace=1, PSD).
-   **Scalability**: Validated the software pipeline up to 12 qubits, confirming the expected exponential growth in classical simulation overhead.

---

## 5. Usage

### Prerequisites
Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Pipeline
**1. Generate Data:**
```bash
python src/data_gen.py --train_size 10000 --shots 1024
```

**2. Train Model:**
```bash
python src/train.py --epochs 20 --hidden_dim 128
```

**3. Validate:**
```bash
python src/validate_benchmarks.py
```
