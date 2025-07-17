import torch
import pytest
from models.experimental.uniad.reference.decoder import DetectionTransformerDecoder
from collections import OrderedDict


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_uniad_decoder(
    device,
    reset_seeds,
):
    weights_path = "/home/ubuntu/punith/tt-metal/models/experimental/uniad/reference/uniad_base_e2e.pth"
    torch_model = DetectionTransformerDecoder(num_layers=6, embed_dim=256, num_heads=8)
    weights = torch.load(weights_path, map_location=torch.device("cpu"))

    prefix = "pts_bbox_head.transformer.decoder"
    filtered = OrderedDict(
        (
            (k[len(prefix) + 1 :], v)  # Remove the prefix from the key
            for k, v in weights["state_dict"].items()
            if k.startswith(prefix)
        )
    )
    # print("filtered",filtered)
    # state_dict=weights["state_dict"]["pts_bbox_head"]#["transformer"]["decoder"]
    torch_model.load_state_dict(filtered)
    torch_model.eval()

    query = torch.load("models/experimental/uniad/reference/decoder_tensors/query_input.pt")
    kwargs = torch.load("models/experimental/uniad/reference/decoder_tensors/kwargs_updated.pt")
    reference_points = torch.load("models/experimental/uniad/reference/decoder_tensors/reference_points_input.pt")
    reg_branches = torch.load("models/experimental/uniad/reference/decoder_tensors/reg_branches_input.pt")

    print("reg_branches", reg_branches)

    output1, output2 = torch_model(
        query=query,
        key=kwargs["key"],
        value=kwargs["value"],
        query_pos=kwargs["query_pos"],
        reference_points=reference_points,
        spatial_shapes=kwargs["spatial_shapes"],
        reg_branches=reg_branches,
        # cls_branches=None,
        # **kwargs,
    )

    # print("torch_model",torch_model)

    gt_intermediate_reference = torch.load(
        "/home/ubuntu/punith/tt-metal/models/experimental/uniad/reference/decoder_tensors/torch.stack(intermediate_reference_points)_output.pt"
    )
    gt_intermediate = torch.load(
        "/home/ubuntu/punith/tt-metal/models/experimental/uniad/reference/decoder_tensors/torch.stack(intermediate)_output.pt"
    )

    print("-------torch output ---------")
    print("gt_intermediate", gt_intermediate)
    print("gt_intermediate_reference", gt_intermediate_reference)

    print("-------reference output ---------")
    print("output_1", output1)
    print("output_2", output2)

    from tests.ttnn.utils_for_testing import assert_with_pcc

    passing, pcc = assert_with_pcc(gt_intermediate, output1, 0.99)
    passing, pcc = assert_with_pcc(gt_intermediate_reference, output2, 0.99)
