import ttnn
import torch
from tests.ttnn.utils_for_testing import assert_with_pcc


def test_div_my_case(device):
    torch_input_tensor_a = torch.load("const_ip.pt")
    torch_input_tensor_b = torch.load("divide_input_1.pt")
    torch_output_tensor = torch.divide(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)
    output = ttnn.divide(input_tensor_a, input_tensor_b)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output_tensor, output)
