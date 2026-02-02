# Final Report: Neural Network Quantum State Tomography

**Date:** 2026-01-15
**Author:** Bollu Venkata Adithya

## Executive Summary
I developed a machine learning model to perform Quantum State Tomography (QST) on a single qubit. The core of this project is a Cholesky-parameterized neural network, which guarantees that every predicted quantum state is physically valid (Hermitian, Positive Semi-Definite, and with Unit Trace). I achieved an average fidelity of **~97.9%** on benchmark states using 1024 simulated measurement shots.

## Methodology

### Model Architecture
I chose a **Cholesky-MLP** architecture for this problem:
- **Input**: 6 measurement probabilities (Expectation values for Z, X, Y bases).
- **Hidden Layers**: 3 layers of size 128 with LeakyReLU activations.
- **Output**: 4 real parameters that form the Cholesky factor L of a 2x2 matrix.
  - I used Softplus to ensure the diagonal elements (L00, L11) are positive.
  - The off-diagonal element (L10) is complex.
- **Reconstruction**: The final density matrix is calculated as ρ = (L L†) / Tr(L L†). This construction enforces the physical constraints by design.

### Dataset and Simulation
- I generated random training states from the Ginibre ensemble to cover both mixed and pure states.
- To simulate real-world conditions, I used 1024 shots per measurement basis (Z, X, Y), applying a multinomial distribution to model statistical noise.

## Results

### Performance Metrics
Evaluation carried out on the validation set of 6 standard states (|0⟩, |1⟩, |+⟩, |-⟩, |i⟩, |-i⟩).

| Metric | Value | Notes |
| :--- | :--- | :--- |
| **Mean Fidelity**       | **0.9792** | High overlap with true states. |
| **Mean Trace Distance** | **0.0311** | Low statistical distance. |
| **Inference Latency**   | **< 1ms**  | Faster than iterative MLE methods |

### Benchmark Breakdown
| State | Fidelity | Trace Distance |
| :--- | :--- | :--- |
| |0⟩ | 0.9852 | 0.0197 |
| |1⟩ | 0.9694 | 0.0349 |
| |+⟩ | 0.9782 | 0.0262 |
| |-⟩ | 0.9842 | 0.0257 |
| |i⟩ | 0.9799 | 0.0380 |
| |-i⟩ | 0.9780 | 0.0419 |

> **Note**: The fidelity is limited primarily by the statistical noise from the 1024 shots (shot noise limit) rather than model capacity.
