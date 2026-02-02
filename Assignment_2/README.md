# One-Qubit Quantum State Tomography (QST)

This project explores a Machine Learning approach to Quantum State Tomography for a single qubit. I built a Multilayer Perceptron (MLP) that learns to reconstruct the density matrix ρ from a set of Pauli measurement probabilities.

## Key Features

- **Physics-Informed**: The model predicts the Cholesky factor L of the density matrix (ρ = L L†). This simple trick guarantees the result is always a valid quantum state (Positive Semi-Definite).
- **Auto-Normalization**: The output is automatically normalized to have Unit Trace.
- **Realistic Data**: I included a script to generate random density matrices and simulate measurement shots using multinomial distributions, mimicking real quantum hardware noise.
- **Validation**: A built-in benchmark script checks performance against standard states (|0⟩, |1⟩, |+⟩, |-⟩, |i⟩, |-i⟩).

## Installation

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *Requires Python 3.8+*

## Usage

### 1. Generate Data
Generate training, testing, and benchmark datasets.
```bash
python src/data_gen.py --train_size 10000 --test_size 1000 --shots 1024
```
Outputs are saved to the `data/` directory.

### 2. Train the Model
Train the Cholesky-MLP on the generated data.
```bash
python src/train.py --epochs 20 --hidden_dim 128
```
The best model is saved to `outputs/best_model.pt`.

### 3. Validate Results
Evaluate the model on the standard benchmark states.
```bash
python src/validate_benchmarks.py
```

## Directory Structure
- `src/`: Source code.
  - `data_gen.py`: Data generation logic.
  - `model.py`: PyTorch model definition (`CholeskyMLP`).
  - `train.py`: Training loop.
  - `hardware_simulation_gen.py`: Hardware simulation VCD generator.
  - `validate_benchmarks.py`: Validation script.
- `data/`: Generated datasets (`.npz`).
- `outputs/`: Saved models and training artifacts.
  - `best_model.pt`: Trained model weights.
  - `hardware_simulation.vcd`: Generated signal trace (Golden Model reference) for hardware verification.

## AI Attribution
For details on the AI tools used in this project, specific prompts, and verification methodology, please refer to [AI_USAGE.md](AI_USAGE.md).
