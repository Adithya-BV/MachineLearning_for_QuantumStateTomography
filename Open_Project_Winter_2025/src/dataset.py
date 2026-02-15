import torch
from torch.utils.data import Dataset
import numpy as np
import pathlib

class QSTDataset(Dataset):
    def __init__(self, data_path: str):
        """
        Args:
            data_path: Path to the .npz file containing 'X' (measurements) and 'y' (density matrices).
        """
        path = pathlib.Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
            
        data = np.load(path)
        self.X = torch.from_numpy(data['X']).float() # Shape: (N, 6)
        
        # We need to process y (density matrices) to be suitable for training if needed,
        # but usually we just keep them as complex matrices for computing loss.
        # However, PyTorch doesn't support complex numbers in all layers/losses easily,
        # so for some losses we might treat real/imag parts separately.
        # For now, let's keep them as (N, 2, 2) complex tensors.
        self.y = torch.from_numpy(data['y']).cfloat() # Shape: (N, 2, 2)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
