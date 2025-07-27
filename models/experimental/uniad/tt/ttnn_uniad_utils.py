# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import math
from torch import Tensor
import numpy as np
from typing import Sequence, Tuple, Union

import ttnn


def pos2posemb2d(pos, num_pos_feats=128, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    device = pos.device()
    dim_t = ttnn.arange(num_pos_feats, device=device)
    dim_t = ttnn.to_layout(dim_t, layout=ttnn.TILE_LAYOUT)
    dim_t = ttnn.div(dim_t, 2)
    dim_t = ttnn.floor(dim_t)

    dim_t = 2 * ttnn.div(dim_t, num_pos_feats)
    dim_t = ttnn.pow(temperature, dim_t)
    pos_x = ttnn.div(ttnn.unsqueeze(pos[..., 0], -1), dim_t)
    pos_y = ttnn.div(ttnn.unsqueeze(pos[..., 1], -1), dim_t)
    pos_x = ttnn.stack((ttnn.sin(pos_x[..., 0::2]), ttnn.cos(pos_x[..., 1::2])), dim=-1)
    if len(pos_x.shape) == 4:
        pos_x = ttnn.reshape(pos_x, (pos_x.shape[0], pos_x.shape[1], pos_x.shape[2] * pos_x.shape[3]))
    elif len(pos_x.shape) == 5:
        pos_x = ttnn.reshape(pos_x, (pos_x.shape[0], pos_x.shape[1], pos_x.shape[2], pos_x.shape[3] * pos_x.shape[4]))
    elif len(pos_x.shape) == 6:
        pos_x = ttnn.reshape(
            pos_x, (pos_x.shape[0], pos_x.shape[1], pos_x.shape[2], pos_x.shape[3], pos_x.shape[4] * pos_x.shape[5])
        )

    pos_y = ttnn.stack((ttnn.sin(pos_y[..., 0::2]), ttnn.cos(pos_y[..., 1::2])), dim=-1)

    if len(pos_y.shape) == 4:
        pos_y = ttnn.reshape(pos_y, (pos_y.shape[0], pos_y.shape[1], pos_y.shape[2] * pos_y.shape[3]))
    elif len(pos_y.shape) == 5:
        pos_y = ttnn.reshape(pos_y, (pos_y.shape[0], pos_y.shape[1], pos_y.shape[2], pos_y.shape[3] * pos_y.shape[4]))
    elif len(pos_y.shape) == 6:
        pos_y = ttnn.reshape(
            pos_y, (pos_y.shape[0], pos_y.shape[1], pos_y.shape[2], pos_y.shape[3], pos_y.shape[4] * pos_y.shape[5])
        )

    posemb = ttnn.concat((pos_y, pos_x), dim=-1)
    return posemb


def norm_points(pos, pc_range):
    x_norm = ttnn.div((ttnn.sub(pos[..., 0], pc_range[0])), (pc_range[3] - pc_range[0]))
    y_norm = ttnn.div((ttnn.sub(pos[..., 1], pc_range[1])), (pc_range[4] - pc_range[1]))
    return ttnn.stack([x_norm, y_norm], dim=-1)


def bivariate_gaussian_activation(ip):
    mu_x = ip[..., 0:1]
    mu_y = ip[..., 1:2]
    sig_x = ip[..., 2:3]
    sig_y = ip[..., 3:4]
    rho = ip[..., 4:5]
    sig_x = ttnn.exp(sig_x)
    sig_y = ttnn.exp(sig_y)
    rho = ttnn.tanh(rho)
    out = ttnn.concat([mu_x, mu_y, sig_x, sig_y, rho], dim=-1)
    return out


def anchor_coordinate_transform(anchors, bbox_results, with_translation_transform=True, with_rotation_transform=True):
    batch_size = len(bbox_results)
    batched_anchors = []
    transformed_anchors = ttnn.unsqueeze(anchors, 0)  # expand num agents: num_groups, num_modes, 12, 2 -> 1, ...
    for i in range(batch_size):
        bboxes, scores, labels, bbox_index, mask = bbox_results[i]
        yaw = bboxes.yaw
        bbox_centers = bboxes.gravity_center
        if with_rotation_transform:
            # TODO(box3d): we have changed yaw to mmdet3d 1.0.0rc6 format, maybe we should change this.
            angle = ttnn.sub(yaw, 3.1415953)  # num_agents, 1
            rot_yaw = rot_2d(angle)  # num_agents, 2, 2
            rot_yaw = ttnn.reshape(
                rot_yaw, (rot_yaw.shape[0], 1, 1, rot_yaw.shape[1], rot_yaw.shape[2])
            )  # num_agents, 1, 1, 2, 2
            transformed_anchors = ttnn.permute(
                transformed_anchors, (0, 1, 2, 4, 3)  # "b g m t c -> b g m c t"
            )  # 1, num_groups, num_modes, 12, 2 -> 1, num_groups, num_modes, 2, 12
            device = rot_yaw.device()
            rot_yaw = ttnn.to_torch(rot_yaw)
            transformed_anchors = ttnn.to_torch(transformed_anchors)
            transformed_anchors = torch.matmul(
                rot_yaw, transformed_anchors
            )  # -> num_agents, num_groups, num_modes, 12, 2
            rot_yaw = ttnn.from_torch(rot_yaw, device=device, layout=ttnn.TILE_LAYOUT)
            transformed_anchors = ttnn.from_torch(transformed_anchors, device=device, layout=ttnn.TILE_LAYOUT)
            transformed_anchors = ttnn.permute(transformed_anchors, (0, 1, 2, 4, 3))  # , "b g m c t -> b g m t c")
        if with_translation_transform:
            transformed_anchors = (
                ttnn.reshape(bbox_centers[:, :2], (bbox_centers[:, :2].shape[0], 1, 1, 1, 2)) + transformed_anchors
            )
        batched_anchors.append(transformed_anchors)
    return ttnn.stack(batched_anchors, dim=0)


def rot_2d(yaw):
    sy, cy = ttnn.sin(yaw), ttnn.cos(yaw)
    out = ttnn.permute(
        ttnn.stack([ttnn.stack([cy, ttnn.mul(sy, -1)], dim=0), ttnn.stack([sy, cy], dim=0)], dim=0), (2, 0, 1)
    )
    return out


def trajectory_coordinate_transform(
    trajectory, bbox_results, with_translation_transform=True, with_rotation_transform=True
):
    batch_size = len(bbox_results)
    batched_trajectories = []
    for i in range(batch_size):
        bboxes, scores, labels, bbox_index, mask = bbox_results[i]
        yaw = bboxes.yaw
        bbox_centers = bboxes.gravity_center
        transformed_trajectory = trajectory[i]
        if with_rotation_transform:
            # we take negtive here, to reverse the trajectory back to ego centric coordinate
            # TODO(box3d): we have changed yaw to mmdet3d 1.0.0rc6 format, maybe we should change this.
            angle = ttnn.mul((ttnn.sub(yaw, 3.1415953)), -1)
            rot_yaw = rot_2d(angle)
            rot_yaw = ttnn.reshape(
                rot_yaw, (rot_yaw.shape[0], 1, 1, rot_yaw.shape[1], rot_yaw.shape[2])
            )  # A, 1, 1, 2, 2
            transformed_trajectory = ttnn.permute(
                transformed_trajectory, (0, 1, 2, 4, 3)  # "a g p t c -> a g p c t"
            )  # A, G, P, 12 ,2 -> # A, G, P, 2, 12

            # Squeezing because matmul doesn't support 5D
            rot_yaw = ttnn.squeeze(rot_yaw, 0)
            transformed_trajectory = ttnn.squeeze(transformed_trajectory, 0)
            device = rot_yaw.device()
            rot_yaw = ttnn.to_torch(rot_yaw)
            transformed_trajectory = ttnn.to_torch(transformed_trajectory)
            transformed_trajectory = torch.matmul(rot_yaw, transformed_trajectory)  # -> A, G, P, 12, 2
            rot_yaw = ttnn.from_torch(rot_yaw, device=device, layout=ttnn.TILE_LAYOUT)
            transformed_trajectory = ttnn.from_torch(transformed_trajectory, device=device, layout=ttnn.TILE_LAYOUT)
            rot_yaw = ttnn.unsqueeze(rot_yaw, 0)
            transformed_trajectory = ttnn.unsqueeze(transformed_trajectory, 0)

            transformed_trajectory = ttnn.permute(transformed_trajectory, (0, 1, 2, 4, 3))  # "a g p c t -> a g p t c"
        if with_translation_transform:
            transformed_trajectory = (
                ttnn.reshape(bbox_centers[:, :2], (bbox_centers[:, :2].shape[0], 1, 1, 1, 2)) + transformed_trajectory
            )
        batched_trajectories.append(transformed_trajectory)
    return ttnn.stack(batched_trajectories, dim=0)


# taken from mmdet3d/structures/bbox_3d/base_box3d.py
class TtBaseInstance3DBoxes:
    """Base class for 3D Boxes.
    Note:
        The box is bottom centered, i.e. the relative position of origin in the
        box is (0.5, 0.5, 0).
    Args:
        tensor (Tensor or np.ndarray or Sequence[Sequence[float]]): The boxes
            data with shape (N, box_dim).
        box_dim (int): Number of the dimension of a box. Each row is
            (x, y, z, x_size, y_size, z_size, yaw). Defaults to 7.
        with_yaw (bool): Whether the box is with yaw rotation. If False, the
            value of yaw will be set to 0 as minmax boxes. Defaults to True.
        origin (Tuple[float]): Relative position of the box origin.
            Defaults to (0.5, 0.5, 0). This will guide the box be converted to
            (0.5, 0.5, 0) mode.
    Attributes:
        tensor (Tensor): Float matrix with shape (N, box_dim).
        box_dim (int): Integer indicating the dimension of a box. Each row is
            (x, y, z, x_size, y_size, z_size, yaw, ...).
        with_yaw (bool): If True, the value of yaw will be set to 0 as minmax
            boxes.
    """

    YAW_AXIS: int = 0

    def __init__(
        self,
        tensor: Union[Tensor, np.ndarray, Sequence[Sequence[float]]],
        box_dim: int = 7,
        with_yaw: bool = True,
        origin: Tuple[float, float, float] = (0.5, 0.5, 0),
    ) -> None:
        # tensor = torch.Tensor(tensor, dtype=ttnn.bfloat16, device=device)
        if tensor.shape[-1] == 0:
            # Use reshape, so we don't end up creating a new tensor that does
            # not depend on the inputs (and consequently confuses jit)
            tensor = tensor.reshape((-1, box_dim))
        assert len(tensor.shape) == 2 and tensor.shape[-1] == box_dim, (
            "The box dimension must be 2 and the length of the last "
            f"dimension must be {box_dim}, but got boxes with shape "
            f"{tensor.shape}."
        )

        if tensor.shape[-1] == 6:
            assert box_dim == 6
            fake_rot = tensor.new_zeros(tensor.shape[0], 1)
            tensor = torch.cat((tensor, fake_rot), dim=-1)
            self.box_dim = box_dim + 1
            self.with_yaw = False
        else:
            self.box_dim = box_dim
            self.with_yaw = with_yaw
        self.tensor = ttnn.clone(tensor)

        if origin != (0.5, 0.5, 0):
            dst = self.tensor.new_tensor((0.5, 0.5, 0))
            src = self.tensor.new_tensor(origin)
            self.tensor[:, :3] += self.tensor[:, 3:6] * (dst - src)

    def __getitem__(self, item: Union[int, slice, np.ndarray, Tensor]) -> "TtBaseInstance3DBoxes":
        original_type = type(self)
        if isinstance(item, int):
            return original_type(self.tensor[item].view(1, -1), box_dim=self.box_dim, with_yaw=self.with_yaw)
        b = self.tensor[item]
        assert b.dim() == 2, f"Indexing on Boxes with {item} failed to return a matrix!"
        return original_type(b, box_dim=self.box_dim, with_yaw=self.with_yaw)

    def to(self, device: Union[str, torch.device], *args, **kwargs) -> "TtBaseInstance3DBoxes":
        original_type = type(self)
        return original_type(self.tensor.to(device, *args, **kwargs), box_dim=self.box_dim, with_yaw=self.with_yaw)

    def gravity_center(self):
        pass

    @property
    def bottom_center(self):
        return self.tensor[:, :3]

    @property
    def yaw(self):
        return self.tensor[:, 6]


# taken from mmdet3d/structures/bbox_3d/lidar_box3d.py, removed all the functions from this class as they are not  invoked
class TtLiDARInstance3DBoxes(TtBaseInstance3DBoxes):
    YAW_AXIS = 2

    @property
    def gravity_center(self):
        bottom_center = self.bottom_center
        gravity_center = ttnn.zeros(bottom_center.shape)  # no use
        gravity_center1 = bottom_center[:, :2]
        gravity_center2 = bottom_center[:, 2:3] + self.tensor[:, 5] * 0.5
        gravity_center = ttnn.concat([gravity_center1, gravity_center2], dim=1)
        return gravity_center
