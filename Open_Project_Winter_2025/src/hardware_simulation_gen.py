import torch
import numpy as np
import pathlib
import argparse
import datetime
from model import CholeskyMLP

def write_vcd_header(f, date_str):
    f.write(f"$date\n   {date_str}\n$end\n")
    f.write("$version\n   PaAC Assignment 2 Reference Model\n$end\n")
    f.write("$timescale\n   1ns\n$end\n")
    f.write("$scope module top $end\n")
    
    # Signals
    # Clock and Reset
    f.write("$var wire 1 ! clk $end\n")
    f.write("$var wire 1 \" rst $end\n")
    
    # Inputs (6 measurement probabilities) - 32 bit float representation
    # We will just dump them as real numbers for simplicity if the viewer supports it, 
    # but standard VCD often wants bit vectors.
    # To be safe and useful for visualizers like GTKWave showing analog, we can use "real".
    
    codes_inputs = ['#', '$', '%', '&', '\'', '(']
    for i in range(6):
        f.write(f"$var real 32 {codes_inputs[i]} in_meas_{i} $end\n")
        
    # Outputs (4 Cholesky params)
    codes_outputs = [')', '*', '+', ',']
    output_names = ['out_L00', 'out_L11', 'out_L10_re', 'out_L10_im']
    for i in range(4):
        f.write(f"$var real 32 {codes_outputs[i]} {output_names[i]} $end\n")
        
    f.write("$upscope $end\n")
    f.write("$enddefinitions $end\n")
    f.write("$dumpvars\n")
    # Initial values
    f.write("0!\n1\"\n") # clk 0, rst 1
    for c in codes_inputs + codes_outputs:
        f.write(f"r0.0 {c}\n")
    f.write("$end\n")
    
    return codes_inputs, codes_outputs

def generate_vcd(args):
    device = torch.device('cpu') # Use CPU for simple generation
    
    # Load Benchmark Data
    data_path = pathlib.Path(args.data_dir) / 'benchmark_data.npz'
    if not data_path.exists():
        print("Benchmark data not found, generating...")
        # Fallback or error? Assuming it exists from previous steps
        return

    data = np.load(data_path)
    X = torch.from_numpy(data['X']).float().to(device)
    
    # Load Model
    model = CholeskyMLP(input_dim=6, hidden_dim=args.hidden_dim).to(device)
    model_path = pathlib.Path(args.output_dir) / 'best_model.pt'
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("Warning: Model not found, using random weights for VCD generation.")
        
    model.eval()
    
    output_vcd_path = pathlib.Path(args.output_dir) / 'hardware_simulation.vcd'
    
    print(f"Generating VCD trace to {output_vcd_path}...")
    
    timestamp = 0
    clk_period = 10 # 10ns clock
    
    with open(output_vcd_path, 'w') as f:
        # Header
        instr_codes, out_codes = write_vcd_header(f, datetime.datetime.now().isoformat())
        
        # Simulation Loop
        with torch.no_grad():
            # Initial Reset
            timestamp += clk_period
            f.write(f"#{timestamp}\n0\"\n") # rst -> 0
            
            for i in range(len(X)):
                inputs = X[i].unsqueeze(0) # (1, 6)
                
                # Forward pass input
                features = model.net(inputs)
                out_raw = model.out_layer(features) # Get raw outputs before softplus for some, but let's strictly follow output of "forward" equivalent logic
                # Actually, let's dump the RAW network outputs (L params) directly as that's what hardware likely outputs
                # But wait, the model code does softplus inside forward.
                # Let's peek at the model logic:
                # l00 = F.softplus(out[:, 0])
                # l11 = F.softplus(out[:, 1])
                # l10_re = out[:, 2]
                # l10_im = out[:, 3]
                
                l00 = torch.nn.functional.softplus(out_raw[:, 0]).item()
                l11 = torch.nn.functional.softplus(out_raw[:, 1]).item()
                l10_re = out_raw[:, 2].item()
                l10_im = out_raw[:, 3].item()
                
                input_vals = inputs[0].tolist()
                output_vals = [l00, l11, l10_re, l10_im]
                
                # Clock High
                f.write(f"#{timestamp}\n1!\n")
                
                # Update inputs and outputs
                for j, val in enumerate(input_vals):
                    f.write(f"r{val:.6f} {instr_codes[j]}\n")
                for j, val in enumerate(output_vals):
                    f.write(f"r{val:.6f} {out_codes[j]}\n")
                    
                timestamp += clk_period // 2
                
                # Clock Low
                f.write(f"#{timestamp}\n0!\n")
                timestamp += clk_period // 2

    print("VCD generation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--hidden_dim', type=int, default=128)
    args = parser.parse_args()
    
    generate_vcd(args)
