import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import argparse
import pathlib
import time
from dataset import QSTDataset
from model import CholeskyMLP

def matrix_mse_loss(rho_pred, rho_true):
    """
    MSE Loss between two density matrices.
    L = Mean(|rho_pred - rho_true|^2)
    """
    # Treat real and imag parts separately for MSE
    diff = rho_pred - rho_true
    return torch.mean(torch.abs(diff)**2)

def fidelity(rho1, rho2):
    """
    Computes fidelity between batch of state vectors or mixed states.
    For pure states (which we mostly have), F = <psi|rho|psi>.
    For full definition: F = (Tr(sqrt(sqrt(rho1) rho2 sqrt(rho1))))^2
    
    Since our ground truth rho2 are pure states (rank 1), 
    F = <psi|rho1|psi> where rho2 = |psi><psi|.
    
    Let's implement the general definition for safety using batch-wise computations,
    or simply use the property that if rho_true is pure: F = Tr(rho_pred * rho_true).
    Wait, if rho_true is pure |psi><psi|, then F(rho_pred, rho_true) = <psi|rho_pred|psi>.
    Tr(rho_pred * rho_true) = Tr(rho_pred * |psi><psi|) = <psi|rho_pred|psi> = F.
    
    So for pure rho_true, Fidelity = real(Tr(rho_pred @ rho_true)).
    """
    # We assume rho_true are pure states from our generator
    # F = Tr(rho1 @ rho2)
    product = torch.matmul(rho1, rho2)
    # Trace: sum of diagonals
    trace = torch.diagonal(product, dim1=-2, dim2=-1).sum(-1)
    return trace.real

def trace_distance(rho1, rho2):
    """
    Trace distance D = 0.5 * Tr|rho1 - rho2|
    Tr|A| = sum of singular values of A.
    """
    diff = rho1 - rho2
    # optimized for 2x2: Singular values can be computed analytically or via torch.linalg.svd
    # torch.linalg.svd is fast enough for 2x2
    try:
        S = torch.linalg.svdvals(diff)
        return 0.5 * S.sum(-1)
    except:
        # Fallback for numerical stability if needed
        return torch.tensor(0.5) 

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Data
    train_ds = QSTDataset(args.data_dir + '/train_data.npz')
    test_ds = QSTDataset(args.data_dir + '/test_data.npz')
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    
    # Model
    model = CholeskyMLP(input_dim=6, hidden_dim=args.hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training Loop
    best_loss = float('inf')
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Starting training...")
    start_time = time.time()
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            
            optimizer.zero_grad()
            y_pred = model(X)
            
            # Loss: 1 - Fidelity is good, but MSE is more convex usually.
            # Let's try MSE first as planned.
            loss = matrix_mse_loss(y_pred, y)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        total_fid = 0
        total_td = 0
        
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                y_pred = model(X)
                
                val_loss += matrix_mse_loss(y_pred, y).item()
                
                # Metrics
                fid = fidelity(y_pred, y)
                td = trace_distance(y_pred, y)
                
                total_fid += fid.sum().item()
                total_td += td.sum().item()
        
        avg_val_loss = val_loss / len(test_loader)
        avg_fid = total_fid / len(test_ds)
        avg_td = total_td / len(test_ds)
        
        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.6f} | Val Loss: {avg_val_loss:.6f} | Val Fid: {avg_fid:.4f} | Val TD: {avg_td:.4f}")
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), output_dir / 'best_model.pt')
            
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nTraining Complete in {total_time:.2f}s")
    print(f"Best Model Saved to {output_dir / 'best_model.pt'}")
    
    # Final Inference Latency Check
    model.eval()
    dummy_input = torch.randn(1, 6).to(device)
    t0 = time.time()
    for _ in range(1000):
        _ = model(dummy_input)
    t1 = time.time()
    latency_ms = (t1 - t0) # total ms for 1000 iter
    print(f"Inference Latency: {latency_ms:.3f} ms/sample")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden_dim', type=int, default=128)
    args = parser.parse_args()
    
    train(args)
