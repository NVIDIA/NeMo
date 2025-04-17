# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Final

import dataverse.utils.alpamayo.ndas_camera_model as ndas_camera_model
import dataverse.utils.alpamayo.rig_decoder as rig_decoder
import dataverse.utils.alpamayo.transformation as transformation
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes

# Used for different predicted trajectories
# It comes from [rgb_to_hex(k) for k in plotly.colors.qualitative.Set3]
TRAJ_MODE_COLOR_MAP: Final[list[str]] = [
    "#8dd3c7",
    "#ffffb3",
    "#bebada",
    "#fb8072",
    "#80b1d3",
    "#fdb462",
    "#b3de69",
    "#fccde5",
    "#d9d9d9",
    "#bc80bd",
    "#ccebc5",
    "#ffed6f",
]


def plot_traj_on_image(
    ax: Axes,
    xyzs: list[np.ndarray],
    camera_intrinsic: ndas_camera_model.FThetaCamera,
    camera_extrinsics: np.ndarray,
    maglev_conf: dict[str, int],
    camera_xyz: np.ndarray = np.zeros(3, dtype=np.float32),
    camera_quat: np.ndarray = np.array([1, 0, 0, 0]),
    colors: list[str] | None = None,
    size: float = 1.0,
):
    """Project trajectories onto 2D image and plot with different colors."""
    num_traj = len(xyzs)
    assert xyzs[0].ndim == 2
    if colors is None:
        colors = [TRAJ_MODE_COLOR_MAP[i % 12] for i in range(num_traj)]
    else:
        assert len(colors) == num_traj

    for i in range(num_traj):
        traj_xyz = xyzs[i]
        uv, _ = transformation.get_traj_proj_on_image(
            traj_xyz=traj_xyz,
            camera_xyz=camera_xyz,
            camera_quat=camera_quat,
            camera_extrinsics=camera_extrinsics,
            camera_intrinsic=camera_intrinsic,
            maglev_conf=maglev_conf,
        )
        ax.scatter(uv[:, 0], uv[:, 1], s=size, c=colors[i])

        # # NOTE we can use following code to draw a patch using car width.
        # # Plot the shadow
        # ax.fill_between(uv[:, 0], uv[:, 1] - 0.2, uv[:, 1] + 0.2, color='blue', alpha=0.3)
        # # Plot the main blue line
        # ax.plot(uv[:, 0], uv[:, 1], color='blue', linewidth=4)
    ax.axis("off")


def render_image_with_traj(
    ax: Axes,
    traj_xyzs: torch.Tensor | None,
    camera_image: torch.Tensor,
    rig_info: dict,
    cam_name: str = "camera_front_wide_120fov",
) -> plt.figure:
    """Plot trajectory on 2D image.

    Args:
        traj_xyzs (torch.Tensor): [K, N, 3], the origin should be on rig at camera_image's
            timestamp.
        camera_image (torch.Tensor): [3, H, W], float in 0-1 range.
        rig_info (str | bytes): file path for rig info or the raw bytes data.
        cam_name (str, optional): camera name used to retrieve camera parameters wrt rig. Defaults
            to "camera_front_wide_120fov".

    Returns:
        plt.figure: figure handler.
    """
    C, H, W = camera_image.shape
    assert C == 3
    assert traj_xyzs.ndim == 3

    rig = rig_info
    info = rig_decoder.decode_rig_info(rig)

    camera_image_np = camera_image.permute(1, 2, 0).numpy()

    camera_extrinsics: torch.Tensor = torch.from_numpy(transformation.sensor_to_rig(info[cam_name]))
    camera_intrinsic = ndas_camera_model.FThetaCamera.from_dict(info[cam_name])

    maglev_conf = transformation.get_video_parameters([cam_name])
    # updating size_h and size_w in maglev_conf in case the inputs image is resize to different
    # ones than what's in maglev.
    maglev_conf["size_h"] = H
    maglev_conf["size_w"] = W

    ax.imshow(camera_image_np)

    plot_traj_on_image(
        ax,
        traj_xyzs.numpy(),
        camera_intrinsic=camera_intrinsic,
        camera_extrinsics=camera_extrinsics,
        maglev_conf=maglev_conf,
    )
