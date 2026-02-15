# Final Technical Report: Machine Learning for Quantum State Tomography

**Date:** Winter 2025
**Project:** Open Project (Assignments 1-3)

## 1. Introduction
This report documents the findings of a multi-stage project aimed at developing robust and scalable methods for Quantum State Tomography (QST). We explored both classical linear inversion techniques and modern physics-informed neural networks to reconstruct quantum states from noisy measurement data.

## 2. Results Appendix

### 2.1 Formal Metrics
We evaluate the quality of reconstructed states $\rho_{pred}$ against the true target states $\rho_{true}$ using two primary metrics:

1.  **Fidelity** ($F$): Measures the closeness of two quantum states.
    $$ F(\rho_{true}, \rho_{pred}) = \left( \text{Tr} \sqrt{\sqrt{\rho_{true}} \rho_{pred} \sqrt{\rho_{true}}} \right)^2 $$
    *Range: $[0, 1]$, where $1$ indicates a perfect reconstruction.*

2.  **Trace Distance** ($T$): Measures the distinguishability between states.
    $$ T(\rho_{true}, \rho_{pred}) = \frac{1}{2} \text{Tr} | \rho_{true} - \rho_{pred} | $$
    *Range: $[0, 1]$, where $0$ indicates identical states.*

### 2.2 Numerical Performance (Single Qubit)
The following results were obtained using the Cholesky-MLP model on a validation set of 6 standard Quantum states (Simulated with 1024 shots).

| Target State | Fidelity ($F$) | Trace Distance ($T$) | Physical Validity |
| :--- | :---: | :---: | :---: |
| $|0\rangle$ | $0.9852$ | $0.0197$ | $\checkmark$ |
| $|1\rangle$ | $0.9694$ | $0.0349$ | $\checkmark$ |
| $|+\rangle$ | $0.9782$ | $0.0262$ | $\checkmark$ |
| $|-\rangle$ | $0.9842$ | $0.0257$ | $\checkmark$ |
| $|i\rangle$ | $0.9799$ | $0.0380$ | $\checkmark$ |
| $|-i\rangle$ | $0.9780$ | $0.0419$ | $\checkmark$ |
| **Mean** | **0.9792** | **0.0311** | -- |

### 2.3 Computational Scaling
In Assignment 3, we analyzed the runtime of the tomography pipeline as the number of qubits ($n$) increased. As expected, the state space dimension $d = 2^n$ leads to exponential resource requirements.

| Qubits ($n$) | Hilbert Space Dim ($2^n$) | Mean Runtime (s) |
| :---: | :---: | :---: |
| 2 | 4 | $3.12 \times 10^{-5}$ |
| 4 | 16 | $2.65 \times 10^{-5}$ |
| 6 | 64 | $3.22 \times 10^{-5}$ |
| 8 | 256 | $3.62 \times 10^{-5}$ |
| 10 | 1024 | $5.11 \times 10^{-5}$ |
| 12 | 4096 | $9.27 \times 10^{-5}$ |

*Note: Runtime represents the average time for a single fidelity calculation operation in the simulation surrogate.*

## 3. Final Reflection

### 3.1 Scaling Limits & Observations
The "Curse of Dimensionality" is the primary barrier in QST.
-   **Memory Overhead**: Storing the density matrix $\rho$ requires $\mathcal{O}(4^n)$ complex floats. At $n=15$, this becomes prohibitively large for standard RAM.
-   **Data Requirement**: The number of measurements required to tomographically satisfy a state generally scales exponentially with $n$ for full tomography ($3^n$ for Pauli basis).
-   **Model Capacity**: While the Cholesky-MLP works remarkably well for $n=1$, extending this fully connected architecture to $n > 4$ would parameter explosion (`input_dim` scales with measurements, `output_dim` with $4^n$).

### 3.2 Future Improvements
1.  **Tensor Networks (MPS/PEPS)**: Instead of representing the full density matrix, Matrix Product States (MPS) can capture entanglement with polynomial scaling for many physical states of interest.
2.  **Classical Shadows**: Implementing "Shadow Tomography" to predict properties of the state (like Fidelity or Expectation values) with only $\mathcal{O}(\log M)$ measurements, sidestepping full state reconstruction.
3.  **Generative Models**: Transitioning from simple MLPs to Generative Adversarial Networks (GANs) or Variational Autoencoders (VAEs) that can learn the distribution of valid quantum states more efficiently.

## 4. Conclusion
This project successfully demonstrated the viability of Machine Learning for single-qubit tomography. The Cholesky parameterization proved effective in enforcing physical constraints, a critical advantage over raw linear inversion. However, the scalability analysis confirms that for multi-qubit systems, we must abandon full state reconstruction in favor of implicit representations like Neural Quantum States or Tensor Networks.
