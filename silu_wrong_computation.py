import torch
import ttnn
import numpy as np
import pandas as pd
from models.utility_functions import comp_pcc

device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))


inputs = np.array([0.10009765625, 0.099609375, 0.09912109375])
inputs_torch = torch.tensor(inputs, dtype=torch.bfloat16)

torch_silu = torch.nn.functional.silu(inputs_torch)
a_tt = ttnn.from_torch(
    inputs_torch, device=device, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT
)
ttnn_silu = ttnn.silu(a_tt)

df = pd.DataFrame(
    {
        "Input Tensor": inputs.tolist(),
        "Torch Silu": torch_silu.tolist(),
        "TTNN Silu": ttnn.to_torch(ttnn_silu).tolist(),
    }
)
pd.set_option("display.precision", 15)
with pd.option_context("display.max_rows", None, "display.precision", 15):
    print(df)
_, pcc = comp_pcc(torch_silu, ttnn.to_torch(ttnn_silu))
print(f"PCC: {pcc}")
