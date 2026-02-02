import torch
import torch.nn as nn
import torch.nn.functional as F

class CholeskyMLP(nn.Module):
    def __init__(self, input_dim: int = 6, hidden_dim: int = 64):
        """
        MLP that predicts the Cholesky factor L of a density matrix rho = L L^dag.
        
        Args:
            input_dim: Number of input measurement probabilities (default 6 for 1-qubit Pauli).
            hidden_dim: Hidden layer size.
        """
        super().__init__()
        
        # Architecture
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.1)
        )
        
        # recursive prediction: 
        # For 1 qubit (2x2 rho), L is lower triangular:
        # [[l00, 0], [l10, l11]]
        # l00, l11 must be real > 0. l10 can be complex.
        # Parameters needed:
        # l00 (1 real), l11 (1 real), l10 (1 complex -> 2 reals).
        # Total = 4 output units.
        self.out_layer = nn.Linear(hidden_dim, 4)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Returns the reconstructed density matrix rho (N, 2, 2)
        """
        features = self.net(x)
        out = self.out_layer(features)
        
        # Decode outputs
        # out: [batch, 4] -> l00_raw, l11_raw, l10_re, l10_im
        
        l00 = F.softplus(out[:, 0]) # Ensure positive real
        l11 = F.softplus(out[:, 1]) # Ensure positive real
        l10_re = out[:, 2]
        l10_im = out[:, 3]
        
        # Construct L
        # L = [[l00, 0], [l10_re + j*l10_im, l11]]
        # We construct this batch-wise.
        
        # Create zero equivalent
        zeros = torch.zeros_like(l00)
        
        # Row 0: [l00, 0]
        row0_re = torch.stack([l00, zeros], dim=1) # (N, 2)
        row0_im = torch.stack([zeros, zeros], dim=1)
        
        # Row 1: [l10_re, l11] (+ im parts)
        row1_re = torch.stack([l10_re, l11], dim=1)
        row1_im = torch.stack([l10_im, zeros], dim=1)
        
        # Stack rows to make (N, 2, 2)
        L_re = torch.stack([row0_re, row1_re], dim=1) # (N, 2, 2)
        L_im = torch.stack([row0_im, row1_im], dim=1)
        
        L = torch.complex(L_re, L_im)
        
        # Compute rho = L @ L^dag
        L_dag = L.mH # Conjugate transpose
        rho_raw = torch.matmul(L, L_dag)
        
        # Normalize Trace to 1
        # Tr(rho) is sum of diagonal elements (which are real because rho is Hermitian)
        # trace = rho_raw[:, 0, 0] + rho_raw[:, 1, 1]...
        # faster: torch.diagonal(rho_raw, dim1=-2, dim2=-1).sum(-1)
        
        trace = torch.diagonal(rho_raw.real, dim1=-2, dim2=-1).sum(-1, keepdim=True)
        trace = trace.unsqueeze(-1) # (N, 1, 1) for broadcasting
        
        rho = rho_raw / (trace + 1e-6) # Add epsilon for stability
        
        return rho
