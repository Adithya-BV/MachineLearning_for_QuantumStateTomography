import numpy as np
import pathlib
import argparse
from typing import Dict, List, Tuple

def get_pauli_projectors() -> Dict[str, Dict[str, np.ndarray]]:
    """Returns Pauli projectors for 1 qubit."""
    # Z-basis
    P0z = np.array([[1, 0], [0, 0]], dtype=complex)
    P1z = np.array([[0, 0], [0, 1]], dtype=complex)
    
    # X-basis
    P0x = 0.5 * np.array([[1, 1], [1, 1]], dtype=complex)
    P1x = 0.5 * np.array([[1, -1], [-1, 1]], dtype=complex)
    
    # Y-basis
    P0y = 0.5 * np.array([[1, -1j], [1j, 1]], dtype=complex)
    P1y = 0.5 * np.array([[1, 1j], [-1j, 1]], dtype=complex)
    
    return {
        'Z': {'0': P0z, '1': P1z},
        'X': {'0': P0x, '1': P1x},
        'Y': {'0': P0y, '1': P1y}
    }

def random_density_matrix(dim: int = 2) -> np.ndarray:
    """Generates a random valid density matrix (Haar random like)."""
    # Ginibre ensemble: G = random complex matrix
    G = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    rho = G @ G.conj().T
    rho /= np.trace(rho)
    return rho

def simulate_measurements(rho: np.ndarray, shots: int = 1024) -> np.ndarray:
    """
    Simulates measurements in X, Y, Z bases.
    Returns array of shape (6,) -> [pZ0, pZ1, pX0, pX1, pY0, pY1] (empirical probabilties)
    """
    projectors = get_pauli_projectors()
    results = []
    
    # Order: Z, X, Y
    for basis in ['Z', 'X', 'Y']:
        probs_basis = []
        outcomes = ['0', '1']
        
        # Calculate true probabilities
        true_probs = []
        for out in outcomes:
            P = projectors[basis][out]
            p = float(np.real(np.trace(rho @ P)))
            p = max(0.0, min(1.0, p)) # Clip
            true_probs.append(p)
        
        # Normalize true probs to sum to 1 to avoid rounding errors
        true_probs = np.array(true_probs)
        true_probs /= true_probs.sum()
        
        # Sample shots
        counts = np.random.multinomial(shots, true_probs)
        empirical_probs = counts / shots
        results.extend(empirical_probs)
        
    return np.array(results)

def generate_dataset(num_samples: int, shots: int, output_file: str):
    print(f"Generating {num_samples} samples with {shots} shots...")
    
    X_data = [] # Measurements
    y_data = [] # Density matrices (flattened or 2x2)
    
    for _ in range(num_samples):
        rho = random_density_matrix()
        meas = simulate_measurements(rho, shots)
        
        X_data.append(meas)
        y_data.append(rho)
        
    X_data = np.array(X_data, dtype=np.float32)
    y_data = np.array(y_data, dtype=np.complex64)
    
    np.savez(output_file, X=X_data, y=y_data)
    print(f"Saved to {output_file}. X shape: {X_data.shape}, y shape: {y_data.shape}")

def generate_benchmark_dataset(shots: int, output_file: str):
    """Generates the 6 specific states from Assignment 1 for validation."""
    print(f"Generating Assignment 1 Benchmark States (|0>, |1>, |+>, |->, |i>, |-i>) with {shots} shots...")
    
    # Define states
    invsqrt2 = 1.0 / np.sqrt(2)
    states = {
        '0': np.array([1, 0], dtype=complex),
        '1': np.array([0, 1], dtype=complex),
        '+': invsqrt2 * np.array([1, 1], dtype=complex),
        '-': invsqrt2 * np.array([1, -1], dtype=complex),
        'i': invsqrt2 * np.array([1, 1j], dtype=complex),
        '-i': invsqrt2 * np.array([1, -1j], dtype=complex)
    }
    
    X_data = []
    y_data = []
    names = []
    
    for name, vec in states.items():
        rho = np.outer(vec, vec.conj())
        meas = simulate_measurements(rho, shots)
        X_data.append(meas)
        y_data.append(rho)
        names.append(name)
        
    X_data = np.array(X_data, dtype=np.float32)
    y_data = np.array(y_data, dtype=np.complex64)
    
    np.savez(output_file, X=X_data, y=y_data, names=names)
    print(f"Saved benchmark to {output_file}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_size', type=int, default=10000)
    parser.add_argument('--test_size', type=int, default=1000)
    parser.add_argument('--shots', type=int, default=1024)
    parser.add_argument('--out_dir', type=str, default='data')
    args = parser.parse_args()
    
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    generate_dataset(args.train_size, args.shots, out_dir / 'train_data.npz')
    generate_dataset(args.test_size, args.shots, out_dir / 'test_data.npz')
    generate_benchmark_dataset(args.shots, out_dir / 'benchmark_data.npz')
