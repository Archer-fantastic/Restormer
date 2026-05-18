print("Script started!")
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from basicsr.models.archs.restormer_arch import Restormer

def test_512():
    print("Setting up model...", flush=True)
    
    device = torch.device('cuda')
    net = Restormer(
        inp_channels=1,
        out_channels=1,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type="BiasFree",
        dual_pixel_task=False
    ).to(device)
    net.train()
    
    print(f"Model created. GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB", flush=True)
    
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4)
    
    print("Testing 384x384...", flush=True)
    x = torch.randn(1, 1, 384, 384).to(device)
    y = net(x)
    loss = y.sum()
    loss.backward()
    optimizer.step()
    print(f"384x384 passed. GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB", flush=True)
    
    optimizer.zero_grad()
    
    print("Testing 512x512...", flush=True)
    try:
        x = torch.randn(1, 1, 512, 512).to(device)
        y = net(x)
        loss = y.sum()
        loss.backward()
        optimizer.step()
        print(f"512x512 passed. GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB", flush=True)
    except Exception as e:
        print(f"Error during 512x512: {e}", flush=True)

if __name__ == '__main__':
    test_512()
