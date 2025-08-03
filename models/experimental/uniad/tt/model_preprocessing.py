# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import torch.nn as nn
from models.experimental.uniad.reference.fpn import FPN
from models.experimental.uniad.reference.resnet import ResNet, ModulatedDeformConv2dPack
from ttnn.model_preprocessing import (
    infer_ttnn_module_args,
    preprocess_model_parameters,
    preprocess_linear_weight,
    preprocess_linear_bias,
    preprocess_layernorm_parameter,
    fold_batch_norm2d_into_conv2d,
)


def custom_preprocessor(model, name):
    parameters = {}

    if isinstance(model, ResNet):
        if isinstance(model, ResNet):
            parameters["res_model"] = {}

        # Initial conv + bn
        weight, bias = fold_batch_norm2d_into_conv2d(model.conv1, model.bn1)
        parameters["res_model"]["conv1"] = {
            "weight": ttnn.from_torch(weight, dtype=ttnn.float32),
            "bias": ttnn.from_torch(bias.reshape((1, 1, 1, -1)), dtype=ttnn.float32),
        }

        # Loop over all layers (layer1 to layer4)
        for layer_idx in range(1, 5):
            layer = getattr(model, f"layer{layer_idx}")
            prefix = f"layer{layer_idx}"  # _{block_idx}"
            parameters["res_model"][prefix] = {}
            for block_idx, block in enumerate(layer):
                parameters["res_model"][prefix][block_idx] = {}

                # conv1, conv2, conv3
                for conv_name in ["conv1", "conv2", "conv3"]:
                    conv = getattr(block, conv_name)
                    if isinstance(conv, ModulatedDeformConv2dPack):
                        parameters["res_model"][prefix][block_idx][conv_name] = {}
                        parameters["res_model"][prefix][block_idx][conv_name]["weight"] = conv.weight
                        parameters["res_model"][prefix][block_idx][conv_name]["bias"] = conv.bias
                        parameters["res_model"][prefix][block_idx][conv_name]["conv_offset"] = {
                            "weight": ttnn.from_torch(conv.conv_offset.weight, dtype=ttnn.float32),
                            "bias": ttnn.from_torch(conv.conv_offset.bias.reshape((1, 1, 1, -1)), dtype=ttnn.float32),
                        }

                        bn = getattr(block, f"bn{conv_name[-1]}")
                        channel_size = bn.num_features

                        # Extract PyTorch tensors
                        weight_torch = bn.weight if bn.affine else None
                        bias_torch = bn.bias if bn.affine else None
                        batch_mean_torch = bn.running_mean
                        batch_var_torch = bn.running_var

                        # Reshape for broadcast compatibility (1, C, 1, 1)
                        batch_mean_torch = batch_mean_torch.view(1, channel_size, 1, 1)
                        batch_var_torch = batch_var_torch.view(1, channel_size, 1, 1)
                        weight_torch = weight_torch.view(1, channel_size, 1, 1) if weight_torch is not None else None
                        bias_torch = bias_torch.view(1, channel_size, 1, 1) if bias_torch is not None else None

                        parameters["res_model"][prefix][block_idx]["bn2"] = {}
                        weight = (
                            ttnn.from_torch(weight_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
                            if weight_torch is not None
                            else None
                        )
                        parameters["res_model"][prefix][block_idx]["bn2"][
                            "weight"
                        ] = weight  # ttnn.to_device(weight, device)

                        bias = (
                            ttnn.from_torch(bias_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
                            if bias_torch is not None
                            else None
                        )
                        parameters["res_model"][prefix][block_idx]["bn2"]["bias"] = bias  # ttnn.to_device(bias, device)

                        running_mean = ttnn.from_torch(batch_mean_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
                        parameters["res_model"][prefix][block_idx]["bn2"][
                            "running_mean"
                        ] = running_mean  # ttnn.to_device(running_mean, device)

                        running_var = ttnn.from_torch(batch_var_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
                        parameters["res_model"][prefix][block_idx]["bn2"]["running_var"] = running_var

                        parameters["res_model"][prefix][block_idx]["bn2"][
                            "eps"
                        ] = bn.eps  # scalar, used directly in ops

                    else:
                        bn = getattr(block, f"bn{conv_name[-1]}")
                        w, b = fold_batch_norm2d_into_conv2d(conv, bn)
                        parameters["res_model"][prefix][block_idx][conv_name] = {
                            "weight": ttnn.from_torch(w, dtype=ttnn.float32),
                            "bias": ttnn.from_torch(b.reshape((1, 1, 1, -1)), dtype=ttnn.float32),
                        }

                # downsample (if present)
                if hasattr(block, "downsample") and block.downsample is not None:
                    ds = block.downsample
                    if isinstance(ds, torch.nn.Sequential):
                        conv = ds[0]
                        bn = ds[1]
                        w, b = fold_batch_norm2d_into_conv2d(conv, bn)
                        parameters["res_model"][prefix][block_idx]["downsample"] = {
                            "weight": ttnn.from_torch(w, dtype=ttnn.float32),
                            "bias": ttnn.from_torch(b.reshape((1, 1, 1, -1)), dtype=ttnn.float32),
                        }

    # if isinstance(model, QueryInteractionModule):
    #     parameters = {
    #     "query_interact": {
    #         "self_attn": {
    #             "out_proj": {
    #                 "weight": preprocess_linear_weight(model.self_attn.out_proj.weight, dtype=ttnn.bfloat16),
    #                 "bias": preprocess_linear_bias(model.self_attn.out_proj.bias, dtype=ttnn.bfloat16),
    #             }
    #         },
    #         "linear1": {
    #             "weight": preprocess_linear_weight(model.linear1.weight, dtype=ttnn.bfloat16),
    #             "bias": preprocess_linear_bias(model.linear1.bias, dtype=ttnn.bfloat16),
    #         },
    #         "linear2": {
    #             "weight": preprocess_linear_weight(model.linear2.weight, dtype=ttnn.bfloat16),
    #             "bias": preprocess_linear_bias(model.linear2.bias, dtype=ttnn.bfloat16),
    #         },
    #         "linear_pos1": {
    #             "weight": preprocess_linear_weight(model.linear_pos1.weight, dtype=ttnn.bfloat16),
    #             "bias": preprocess_linear_bias(model.linear_pos1.bias, dtype=ttnn.bfloat16),
    #         },
    #         "linear_pos2": {
    #             "weight": preprocess_linear_weight(model.linear_pos2.weight, dtype=ttnn.bfloat16),
    #             "bias": preprocess_linear_bias(model.linear_pos2.bias, dtype=ttnn.bfloat16),
    #         },
    #         "linear_feat1": {
    #             "weight": preprocess_linear_weight(model.linear_feat1.weight, dtype=ttnn.bfloat16),
    #             "bias": preprocess_linear_bias(model.linear_feat1.bias, dtype=ttnn.bfloat16),
    #         },
    #         "linear_feat2": {
    #             "weight": preprocess_linear_weight(model.linear_feat2.weight, dtype=ttnn.bfloat16),
    #             "bias": preprocess_linear_bias(model.linear_feat2.bias, dtype=ttnn.bfloat16),
    #         },
    #         "norm_pos": preprocess_layernorm(model.norm_pos, dtype=ttnn.bfloat16),
    #         "norm_feat": preprocess_layernorm(model.norm_feat, dtype=ttnn.bfloat16),
    #         "norm1": preprocess_layernorm(model.norm1, dtype=ttnn.bfloat16),
    #         "norm2": preprocess_layernorm(model.norm2, dtype=ttnn.bfloat16),
    #     }
    # }

    if isinstance(model, FPN):
        parameters["fpn"] = {}

        # Lateral Convs
        parameters["fpn"]["lateral_convs"] = {}

        parameters["fpn"]["lateral_convs"]["0"] = {}
        parameters["fpn"]["lateral_convs"]["0"]["conv"] = {}
        parameters["fpn"]["lateral_convs"]["0"]["conv"]["weight"] = ttnn.from_torch(
            model.lateral_convs[0].conv.weight, dtype=ttnn.float32
        )
        bias = model.lateral_convs[0].conv.bias.reshape((1, 1, 1, -1))
        parameters["fpn"]["lateral_convs"]["0"]["conv"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)
        parameters["fpn"]["lateral_convs"]["0"]["conv"]["height"] = 80
        parameters["fpn"]["lateral_convs"]["0"]["conv"]["width"] = 45
        parameters["fpn"]["lateral_convs"]["0"]["conv"]["batch"] = 6

        parameters["fpn"]["lateral_convs"]["1"] = {}
        parameters["fpn"]["lateral_convs"]["1"]["conv"] = {}
        parameters["fpn"]["lateral_convs"]["1"]["conv"]["weight"] = ttnn.from_torch(
            model.lateral_convs[1].conv.weight, dtype=ttnn.float32
        )
        bias = model.lateral_convs[1].conv.bias.reshape((1, 1, 1, -1))
        parameters["fpn"]["lateral_convs"]["1"]["conv"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)
        parameters["fpn"]["lateral_convs"]["1"]["conv"]["height"] = 40
        parameters["fpn"]["lateral_convs"]["1"]["conv"]["width"] = 23
        parameters["fpn"]["lateral_convs"]["1"]["conv"]["batch"] = 6

        parameters["fpn"]["lateral_convs"]["2"] = {}
        parameters["fpn"]["lateral_convs"]["2"]["conv"] = {}
        parameters["fpn"]["lateral_convs"]["2"]["conv"]["weight"] = ttnn.from_torch(
            model.lateral_convs[2].conv.weight, dtype=ttnn.float32
        )
        bias = model.lateral_convs[2].conv.bias.reshape((1, 1, 1, -1))
        parameters["fpn"]["lateral_convs"]["2"]["conv"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)
        parameters["fpn"]["lateral_convs"]["2"]["conv"]["height"] = 20
        parameters["fpn"]["lateral_convs"]["2"]["conv"]["width"] = 12
        parameters["fpn"]["lateral_convs"]["2"]["conv"]["batch"] = 6
        # FPN Convs
        parameters["fpn"]["fpn_convs"] = {}

        parameters["fpn"]["fpn_convs"]["0"] = {}
        parameters["fpn"]["fpn_convs"]["0"]["conv"] = {}
        parameters["fpn"]["fpn_convs"]["0"]["conv"]["weight"] = ttnn.from_torch(
            model.fpn_convs[0].conv.weight, dtype=ttnn.float32
        )
        bias = model.fpn_convs[0].conv.bias.reshape((1, 1, 1, -1))
        parameters["fpn"]["fpn_convs"]["0"]["conv"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)
        parameters["fpn"]["fpn_convs"]["0"]["conv"]["height"] = 80
        parameters["fpn"]["fpn_convs"]["0"]["conv"]["width"] = 45
        parameters["fpn"]["fpn_convs"]["0"]["conv"]["batch"] = 6

        parameters["fpn"]["fpn_convs"]["1"] = {}
        parameters["fpn"]["fpn_convs"]["1"]["conv"] = {}
        parameters["fpn"]["fpn_convs"]["1"]["conv"]["weight"] = ttnn.from_torch(
            model.fpn_convs[1].conv.weight, dtype=ttnn.float32
        )
        bias = model.fpn_convs[1].conv.bias.reshape((1, 1, 1, -1))
        parameters["fpn"]["fpn_convs"]["1"]["conv"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)
        parameters["fpn"]["fpn_convs"]["1"]["conv"]["height"] = 40
        parameters["fpn"]["fpn_convs"]["1"]["conv"]["width"] = 23
        parameters["fpn"]["fpn_convs"]["1"]["conv"]["batch"] = 6

        parameters["fpn"]["fpn_convs"]["2"] = {}
        parameters["fpn"]["fpn_convs"]["2"]["conv"] = {}
        parameters["fpn"]["fpn_convs"]["2"]["conv"]["weight"] = ttnn.from_torch(
            model.fpn_convs[2].conv.weight, dtype=ttnn.float32
        )
        bias = model.fpn_convs[2].conv.bias.reshape((1, 1, 1, -1))
        parameters["fpn"]["fpn_convs"]["2"]["conv"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)
        parameters["fpn"]["fpn_convs"]["2"]["conv"]["height"] = 20
        parameters["fpn"]["fpn_convs"]["2"]["conv"]["width"] = 12
        parameters["fpn"]["fpn_convs"]["2"]["conv"]["batch"] = 6

        parameters["fpn"]["fpn_convs"]["3"] = {}
        parameters["fpn"]["fpn_convs"]["3"]["conv"] = {}
        parameters["fpn"]["fpn_convs"]["3"]["conv"]["weight"] = ttnn.from_torch(
            model.fpn_convs[3].conv.weight, dtype=ttnn.float32
        )
        bias = model.fpn_convs[3].conv.bias.reshape((1, 1, 1, -1))
        parameters["fpn"]["fpn_convs"]["3"]["conv"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)
        parameters["fpn"]["fpn_convs"]["3"]["conv"]["height"] = 20
        parameters["fpn"]["fpn_convs"]["3"]["conv"]["width"] = 12
        parameters["fpn"]["fpn_convs"]["3"]["conv"]["batch"] = 6

        # if isinstance(model, TemporalSelfAttention):
        #     parameters["temporal_self_attention"] = {}
        #     parameters["temporal_self_attention"]["sampling_offsets"] = {}
        #     parameters["temporal_self_attention"]["sampling_offsets"]["weight"] = preprocess_linear_weight(
        #         model.sampling_offsets.weight, dtype=ttnn.bfloat16
        #     )
        #     parameters["temporal_self_attention"]["sampling_offsets"]["bias"] = preprocess_linear_bias(
        #         model.sampling_offsets.bias, dtype=ttnn.bfloat16
        #     )
        #     parameters["temporal_self_attention"]["attention_weights"] = {}
        #     parameters["temporal_self_attention"]["attention_weights"]["weight"] = preprocess_linear_weight(
        #         model.attention_weights.weight, dtype=ttnn.bfloat16
        #     )
        #     parameters["temporal_self_attention"]["attention_weights"]["bias"] = preprocess_linear_bias(
        #         model.attention_weights.bias, dtype=ttnn.bfloat16
        #     )
        #     parameters["temporal_self_attention"]["value_proj"] = {}
        #     parameters["temporal_self_attention"]["value_proj"]["weight"] = preprocess_linear_weight(
        #         model.value_proj.weight, dtype=ttnn.bfloat16
        #     )
        #     parameters["temporal_self_attention"]["value_proj"]["bias"] = preprocess_linear_bias(
        #         model.value_proj.bias, dtype=ttnn.bfloat16
        #     )
        #     parameters["temporal_self_attention"]["output_proj"] = {}
        #     parameters["temporal_self_attention"]["output_proj"]["weight"] = preprocess_linear_weight(
        #         model.output_proj.weight, dtype=ttnn.bfloat16
        #     )
        #     parameters["temporal_self_attention"]["output_proj"]["bias"] = preprocess_linear_bias(
        #         model.output_proj.bias, dtype=ttnn.bfloat16
        #     )
        # # if isinstance(model, MultiheadAttention):
        # #     parameters = {
        # #         "attentions": {},
        # #     }
        # #     parameters["attentions"][f"attn{0}"] = {
        # #         "in_proj": {
        # #             "weight": preprocess_linear_weight(model.attn.in_proj_weight, dtype=ttnn.bfloat16),
        # #             "bias": preprocess_linear_bias(model.attn.in_proj_bias, dtype=ttnn.bfloat16),
        # #         },
        # #         "out_proj": {
        # #             "weight": preprocess_linear_weight(model.attn.out_proj.weight, dtype=ttnn.bfloat16),
        # #             "bias": preprocess_linear_bias(model.attn.out_proj.bias, dtype=ttnn.bfloat16),
        # #         },
        # #     }
        # if isinstance(model, FFN):
        #     parameters = {
        #         "ffn": {},
        #     }
        #     parameters["ffn"][f"ffn0"] = {
        #         "linear1": {
        #             "weight": preprocess_linear_weight(model.layers[0][0].weight, dtype=ttnn.bfloat16),
        #             "bias": preprocess_linear_bias(model.layers[0][0].bias, dtype=ttnn.bfloat16),
        #         },
        #         "linear2": {
        #             "weight": preprocess_linear_weight(model.layers[1].weight, dtype=ttnn.bfloat16),
        #             "bias": preprocess_linear_bias(model.layers[1].bias, dtype=ttnn.bfloat16),
        #         },
        #     }

        # if isinstance(model, CustomMSDeformableAttention):
        #     parameters = {
        #         "attentions": {},
        #     }
        #     parameters["attentions"][f"attn1"] = {
        #         "sampling_offsets": {
        #             "weight": preprocess_linear_weight(model.sampling_offsets.weight, dtype=ttnn.bfloat16),
        #             "bias": preprocess_linear_bias(model.sampling_offsets.bias, dtype=ttnn.bfloat16),
        #         },
        #         "attention_weights": {
        #             "weight": preprocess_linear_weight(model.attention_weights.weight, dtype=ttnn.bfloat16),
        #             "bias": preprocess_linear_bias(model.attention_weights.bias, dtype=ttnn.bfloat16),
        #         },
        #         "value_proj": {
        #             "weight": preprocess_linear_weight(model.value_proj.weight, dtype=ttnn.bfloat16),
        #             "bias": preprocess_linear_bias(model.value_proj.bias, dtype=ttnn.bfloat16),
        #         },
        #         "output_proj": {
        #             "weight": preprocess_linear_weight(model.output_proj.weight, dtype=ttnn.bfloat16),
        #             "bias": preprocess_linear_bias(model.output_proj.bias, dtype=ttnn.bfloat16),
        #         },
        #     }

        # if isinstance(model, BEVFormerEncoder):
        #     parameters = {"layers": {}}

        #     for i, layer in enumerate(model.layers):  # BaseTransformerLayer
        #         layer_dict = {
        #             "attentions": {},
        #             "ffn": {},
        #             "norms": {},
        #         }

        #         # Norms
        #         for n, norm in enumerate(layer.norms):
        #             if isinstance(norm, nn.LayerNorm):
        #                 layer_dict["norms"][f"norm{n}"] = {
        #                     "weight": preprocess_layernorm_parameter(norm.weight, dtype=ttnn.bfloat16),
        #                     "bias": preprocess_layernorm_parameter(norm.bias, dtype=ttnn.bfloat16),
        #                 }

        #         # FFNs
        #         for k, ffn in enumerate(layer.ffns):
        #             layer_dict["ffn"][f"ffn{k}"] = {
        #                 "linear1": {
        #                     "weight": preprocess_linear_weight(ffn.layers[0][0].weight, dtype=ttnn.bfloat16),
        #                     "bias": preprocess_linear_bias(ffn.layers[0][0].bias, dtype=ttnn.bfloat16),
        #                 },
        #                 "linear2": {
        #                     "weight": preprocess_linear_weight(ffn.layers[1].weight, dtype=ttnn.bfloat16),
        #                     "bias": preprocess_linear_bias(ffn.layers[1].bias, dtype=ttnn.bfloat16),
        #                 },
        #             }

        #         # Attentions
        #         for j, attn in enumerate(layer.attentions):
        #             if isinstance(attn, TemporalSelfAttention):
        #                 layer_dict["attentions"][f"attn{j}"] = {
        #                     "sampling_offsets": {
        #                         "weight": preprocess_linear_weight(attn.sampling_offsets.weight, dtype=ttnn.bfloat16),
        #                         "bias": preprocess_linear_bias(attn.sampling_offsets.bias, dtype=ttnn.bfloat16),
        #                     },
        #                     "attention_weights": {
        #                         "weight": preprocess_linear_weight(attn.attention_weights.weight, dtype=ttnn.bfloat16),
        #                         "bias": preprocess_linear_bias(attn.attention_weights.bias, dtype=ttnn.bfloat16),
        #                     },
        #                     "value_proj": {
        #                         "weight": preprocess_linear_weight(attn.value_proj.weight, dtype=ttnn.bfloat16),
        #                         "bias": preprocess_linear_bias(attn.value_proj.bias, dtype=ttnn.bfloat16),
        #                     },
        #                     "output_proj": {
        #                         "weight": preprocess_linear_weight(attn.output_proj.weight, dtype=ttnn.bfloat16),
        #                         "bias": preprocess_linear_bias(attn.output_proj.bias, dtype=ttnn.bfloat16),
        #                     },
        #                 }

        #             elif isinstance(attn, SpatialCrossAttention):
        #                 layer_dict["attentions"][f"attn{j}"] = {
        #                     "sampling_offsets": {
        #                         "weight": preprocess_linear_weight(
        #                             attn.deformable_attention.sampling_offsets.weight, dtype=ttnn.bfloat16
        #                         ),
        #                         "bias": preprocess_linear_bias(
        #                             attn.deformable_attention.sampling_offsets.bias, dtype=ttnn.bfloat16
        #                         ),
        #                     },
        #                     "attention_weights": {
        #                         "weight": preprocess_linear_weight(
        #                             attn.deformable_attention.attention_weights.weight, dtype=ttnn.bfloat16
        #                         ),
        #                         "bias": preprocess_linear_bias(
        #                             attn.deformable_attention.attention_weights.bias, dtype=ttnn.bfloat16
        #                         ),
        #                     },
        #                     "value_proj": {
        #                         "weight": preprocess_linear_weight(
        #                             attn.deformable_attention.value_proj.weight, dtype=ttnn.bfloat16
        #                         ),
        #                         "bias": preprocess_linear_bias(
        #                             attn.deformable_attention.value_proj.bias, dtype=ttnn.bfloat16
        #                         ),
        #                     },
        #                     "output_proj": {
        #                         "weight": preprocess_linear_weight(attn.output_proj.weight, dtype=ttnn.bfloat16),
        #                         "bias": preprocess_linear_bias(attn.output_proj.bias, dtype=ttnn.bfloat16),
        #                     },
        #                 }

        #         parameters["layers"][f"layer{i}"] = layer_dict

        # if isinstance(model, MapDetectionTransformerDecoder or DetectionTransformerDecoder):
        #     parameters = {"layers": {}}

        #     for i, layer in enumerate(model.layers):  # BaseTransformerLayer
        #         layer_dict = {
        #             "attentions": {},
        #             "ffn": {},
        #             "norms": {},
        #         }

        #         # Norms
        #         for n, norm in enumerate(layer.norms):
        #             if isinstance(norm, nn.LayerNorm):
        #                 layer_dict["norms"][f"norm{n}"] = {
        #                     "weight": preprocess_layernorm_parameter(norm.weight, dtype=ttnn.bfloat16),
        #                     "bias": preprocess_layernorm_parameter(norm.bias, dtype=ttnn.bfloat16),
        #                 }

        #         # FFNs
        #         for k, ffn in enumerate(layer.ffns):
        #             layer_dict["ffn"][f"ffn{k}"] = {
        #                 "linear1": {
        #                     "weight": preprocess_linear_weight(ffn.layers[0][0].weight, dtype=ttnn.bfloat16),
        #                     "bias": preprocess_linear_bias(ffn.layers[0][0].bias, dtype=ttnn.bfloat16),
        #                 },
        #                 "linear2": {
        #                     "weight": preprocess_linear_weight(ffn.layers[1].weight, dtype=ttnn.bfloat16),
        #                     "bias": preprocess_linear_bias(ffn.layers[1].bias, dtype=ttnn.bfloat16),
        #                 },
        #             }

        #         # Attentions
        #         for j, attn in enumerate(layer.attentions):
        #             if isinstance(attn, MultiheadAttention):
        #                 layer_dict["attentions"][f"attn{j}"] = {
        #                     "in_proj": {
        #                         "weight": preprocess_linear_weight(attn.attn.in_proj_weight, dtype=ttnn.bfloat16),
        #                         "bias": preprocess_linear_bias(attn.attn.in_proj_bias, dtype=ttnn.bfloat16),
        #                     },
        #                     "out_proj": {
        #                         "weight": preprocess_linear_weight(attn.attn.out_proj.weight, dtype=ttnn.bfloat16),
        #                         "bias": preprocess_linear_bias(attn.attn.out_proj.bias, dtype=ttnn.bfloat16),
        #                     },
        #                 }

        #             elif isinstance(attn, CustomMSDeformableAttention):
        #                 layer_dict["attentions"][f"attn{j}"] = {
        #                     "sampling_offsets": {
        #                         "weight": preprocess_linear_weight(attn.sampling_offsets.weight, dtype=ttnn.bfloat16),
        #                         "bias": preprocess_linear_bias(attn.sampling_offsets.bias, dtype=ttnn.bfloat16),
        #                     },
        #                     "attention_weights": {
        #                         "weight": preprocess_linear_weight(attn.attention_weights.weight, dtype=ttnn.bfloat16),
        #                         "bias": preprocess_linear_bias(attn.attention_weights.bias, dtype=ttnn.bfloat16),
        #                     },
        #                     "value_proj": {
        #                         "weight": preprocess_linear_weight(attn.value_proj.weight, dtype=ttnn.bfloat16),
        #                         "bias": preprocess_linear_bias(attn.value_proj.bias, dtype=ttnn.bfloat16),
        #                     },
        #                     "output_proj": {
        #                         "weight": preprocess_linear_weight(attn.output_proj.weight, dtype=ttnn.bfloat16),
        #                         "bias": preprocess_linear_bias(attn.output_proj.bias, dtype=ttnn.bfloat16),
        #                     },
        #                 }

        #         parameters["layers"][f"layer{i}"] = layer_dict

        return parameters

    return parameters


def create_uniad_FPN_parameters(model, input_tensors, device=None):
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )
    parameters.conv_args = {}
    parameters.conv_args = infer_ttnn_module_args(
        model=model, run_model=lambda model: model(input_tensors), device=device
    )
    parameters["model_args"] = model
    # assert parameters is not None
    # for key in parameters.conv_args.keys():
    #     parameters.conv_args[key].module = getattr(model, key)
    return parameters


def create_uniad_model_parameters_sca(model, input_tensor, device=None):
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )
    parameters.conv_args = {}
    parameters.conv_args = infer_ttnn_module_args(
        model=model,
        run_model=lambda model: model(
            input_tensor[0],
            key=input_tensor[1],
            value=input_tensor[2],
            reference_points=input_tensor[3],
            spatial_shapes=input_tensor[4],
            reference_points_cam=input_tensor[5],
            bev_mask=input_tensor[6],
            level_start_index=input_tensor[7],
        ),
        device=None,
    )
    assert parameters is not None
    for key in parameters.conv_args.keys():
        parameters.conv_args[key].module = getattr(model, key)
    return parameters


def create_uniad_model_parameters_tsa(model, input_tensor, device=None):
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )
    parameters.conv_args = {}
    parameters.conv_args = infer_ttnn_module_args(
        model=model,
        run_model=lambda model: model(
            input_tensor[0],
            query_pos=input_tensor[1],
            reference_points=input_tensor[2],
            spatial_shapes=input_tensor[3],
            level_start_index=input_tensor[4],
        ),
        device=None,
    )
    assert parameters is not None
    for key in parameters.conv_args.keys():
        parameters.conv_args[key].module = getattr(model, key)
    return parameters


# def create_uniad_model_parameters_decoder(model: ResNet, input_tensor, device=None):
#     parameters = preprocess_model_parameters(
#         initialize_model=lambda: model,
#         custom_preprocessor=custom_preprocessor,
#         device=device,
#     )
#     return parameters


def create_uniad_model_parameters_encoder(model, device=None):
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )
    return parameters


import ttnn
import torch.nn as nn

from ttnn.model_preprocessing import (
    preprocess_linear_weight,
    preprocess_linear_bias,
    preprocess_layernorm_parameter,
)


def extract_sequential_branch(module_list, dtype, device):
    branch_params = {}

    for i, mod in enumerate(module_list):
        layer_params = {}
        layer_index = 0

        if isinstance(mod, nn.Sequential):
            layers = mod
        elif hasattr(mod, "mlp") and isinstance(mod.mlp, nn.Sequential):
            layers = mod.mlp
        else:
            layers = [mod]

        for layer in layers:
            if isinstance(layer, nn.Linear):
                layer_params[str(layer_index)] = {
                    "weight": ttnn.to_device(preprocess_linear_weight(layer.weight, dtype=dtype), device=device),
                    "bias": ttnn.to_device(preprocess_linear_bias(layer.bias, dtype=dtype), device=device),
                }
                layer_index += 1
            elif isinstance(layer, nn.LayerNorm):
                layer_params[f"{layer_index}_norm"] = {
                    "weight": preprocess_layernorm_parameter(layer.weight, dtype=dtype),
                    "bias": preprocess_layernorm_parameter(layer.bias, dtype=dtype),
                }
                layer_index += 1

        branch_params[str(i)] = layer_params

    return branch_params


def create_uniad_model_resnet(model: ResNet, input_tensor, device=None):
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )
    parameters.conv_args = {}
    parameters.conv_args = infer_ttnn_module_args(model=model, run_model=lambda model: model(input_tensor), device=None)
    assert parameters is not None
    for key in parameters.conv_args.keys():
        parameters.conv_args[key].module = getattr(model, key)
    return parameters


def create_uniad_model_parameters(model, input_tensor, device=None):
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )
    parameters.conv_args = {"img_backbone": {}, "img_neck": {}}

    parameters.conv_args["img_backbone"] = infer_ttnn_module_args(
        model=model.img_backbone,
        run_model=lambda model: model(input_tensor),
        device=None,
    )

    for key in parameters.conv_args.keys():
        if key == "img_backbone":
            for conv_key in parameters.conv_args[key].keys():
                parameters.conv_args[key][conv_key].module = getattr(model.img_backbone, conv_key)

    parameters["img_neck"]["model_args"] = model.img_neck

    return parameters
