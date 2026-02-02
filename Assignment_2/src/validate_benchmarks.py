import torch
import numpy as np
import argparse
import pathlib
from model import CholeskyMLP
from dataset import QSTDataset
from train import fidelity, trace_distance

def validate_benchmarks(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Benchmark Data
    data_path = pathlib.Path(args.data_dir) / 'benchmark_data.npz'
    if not data_path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {data_path}")
    
    data = np.load(data_path)
    X = torch.from_numpy(data['X']).float().to(device)
    y_true = torch.from_numpy(data['y']).cfloat().to(device)
    names = data['names']
    
    # Load Model
    model = CholeskyMLP(input_dim=6, hidden_dim=args.hidden_dim).to(device)
    model.load_state_dict(torch.load(pathlib.Path(args.output_dir) / 'best_model.pt', map_location=device))
    model.eval()
    
    print("\n=== Validating on Assignment 1 Benchmark States ===")
    print(f"{'State':<10} {'Fidelity':<10} {'TraceDist':<10} {'Status'}")
    print("-" * 50)
    
    with torch.no_grad():
        y_pred = model(X)
        
        # Calculate metrics per sample
        fidelities = []
        trace_dists = []
        
        for i in range(len(names)):
            # Per-sample metric
            f_i = fidelity(y_pred[i:i+1], y_true[i:i+1]).item()
            td_i = trace_distance(y_pred[i:i+1], y_true[i:i+1]).item()
            
            fidelities.append(f_i)
            trace_dists.append(td_i)
            
            status = "PASS" if f_i > 0.99 else "WARN"
            print(f"{names[i]:<10} {f_i:.4f}     {td_i:.4f}     {status}")

    print("-" * 50)
    print(f"Mean Fidelity: {np.mean(fidelities):.4f}")
    print(f"Mean Trace Distance: {np.mean(trace_dists):.4f}")
    
    if np.mean(fidelities) > 0.99:
        print("\nSUCCESS: Model matches Assignment 1 Linear Inversion performance (>0.99)!")
    else:
        print("\nNOTE: Performance slightly lower than Linear Inversion (expected due to NN approximation).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--hidden_dim', type=int, default=128)
    args = parser.parse_args()
    
    validate_benchmarks(args)
