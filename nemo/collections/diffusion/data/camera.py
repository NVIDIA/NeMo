# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

# pylint: disable=C0115,C0116,C0301

import numpy as np
import torch


class Pose:
    """
    A class of operations on camera poses (PyTorch tensors with shape [...,3,4]).
    Each [3,4] camera pose takes the form of [R|t].
    """

    def __call__(self, R=None, t=None):
        # Construct a camera pose from the given R and/or t.
        assert R is not None or t is not None
        if R is None:
            if not isinstance(t, torch.Tensor):
                t = torch.tensor(t)
            R = torch.eye(3, device=t.device).repeat(*t.shape[:-1], 1, 1)
        elif t is None:
            if not isinstance(R, torch.Tensor):
                R = torch.tensor(R)
            t = torch.zeros(R.shape[:-1], device=R.device)
        else:
            if not isinstance(R, torch.Tensor):
                R = torch.tensor(R)
            if not isinstance(t, torch.Tensor):
                t = torch.tensor(t)
        assert R.shape[:-1] == t.shape and R.shape[-2:] == (3, 3)
        R = R.float()
        t = t.float()
        pose = torch.cat([R, t[..., None]], dim=-1)  # [...,3,4]
        assert pose.shape[-2:] == (3, 4)
        return pose

    def invert(self, pose, use_inverse=False):
        # Invert a camera pose.
        R, t = pose[..., :3], pose[..., 3:]
        R_inv = R.inverse() if use_inverse else R.transpose(-1, -2)
        t_inv = (-R_inv @ t)[..., 0]
        pose_inv = self(R=R_inv, t=t_inv)
        return pose_inv

    def compose(self, pose_list):
        # Compose a sequence of poses together.
        # pose_new(x) = poseN o ... o pose2 o pose1(x)
        pose_new = pose_list[0]
        for pose in pose_list[1:]:
            pose_new = self.compose_pair(pose_new, pose)
        return pose_new

    def compose_pair(self, pose_a, pose_b):
        # pose_new(x) = pose_b o pose_a(x)
        R_a, t_a = pose_a[..., :3], pose_a[..., 3:]
        R_b, t_b = pose_b[..., :3], pose_b[..., 3:]
        R_new = R_b @ R_a
        t_new = (R_b @ t_a + t_b)[..., 0]
        pose_new = self(R=R_new, t=t_new)
        return pose_new

    def scale_center(self, pose, scale):
        """Scale the camera center from the origin.
        0 = R@c+t --> c = -R^T@t (camera center in world coordinates)
        0 = R@(sc)+t' --> t' = -R@(sc) = -R@(-R^T@st) = st
        """
        R, t = pose[..., :3], pose[..., 3:]
        pose_new = torch.cat([R, t * scale], dim=-1)
        return pose_new

    def interpolate(self, pose_a, pose_b, alpha):
        """Interpolate between two poses with Slerp.
        Args:
            pose_a (tensor [...,3,4]): Pose at time t=0.
            pose_b (tensor [...,3,4]): Pose at time t=1.
            alpha (tensor [...,1]): Interpolation parameter.
        Returns:
            pose (tensor [...,3,4]): Pose at time t.
        """
        R_a, t_a = pose_a[..., :3], pose_a[..., 3:]
        R_b, t_b = pose_b[..., :3], pose_b[..., 3:]
        q_a = quaternion.R_to_q(R_a)  # [...,4]
        q_b = quaternion.R_to_q(R_b)  # [...,4]
        q_intp = quaternion.interpolate(q_a, q_b, alpha)  # [...,4]
        R_intp = quaternion.q_to_R(q_intp)  # [...,3,3]
        t_intp = (1 - alpha) * t_a + alpha * t_b  # [...,3]
        pose_intp = torch.cat([R_intp, t_intp], dim=-1)  # [...,3,4]
        return pose_intp

    def to_4x4(self, pose):
        last_row = torch.tensor([0, 0, 0, 1], device=pose.device)[None, None].expand(pose.shape[0], 1, 4)
        return torch.cat([pose, last_row], dim=-2)


class Lie:
    """
    Lie algebra for SO(3) and SE(3) operations in PyTorch.
    """

    def so3_to_SO3(self, w):  # [..., 3]
        wx = self.skew_symmetric(w)
        theta = w.norm(dim=-1)[..., None, None]
        eye = torch.eye(3, device=w.device, dtype=torch.float32)
        A = self.taylor_A(theta)
        B = self.taylor_B(theta)
        R = eye + A * wx + B * wx @ wx
        return R

    def SO3_to_so3(self, R, eps=1e-7):  # [..., 3, 3]
        trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
        theta = ((trace - 1) / 2).clamp(-1 + eps, 1 - eps).acos_()[
            ..., None, None
        ] % np.pi  # ln(R) will explode if theta==pi
        lnR = 1 / (2 * self.taylor_A(theta) + 1e-8) * (R - R.transpose(-2, -1))  # FIXME: wei-chiu finds it weird
        w0, w1, w2 = lnR[..., 2, 1], lnR[..., 0, 2], lnR[..., 1, 0]
        w = torch.stack([w0, w1, w2], dim=-1)
        return w

    def se3_to_SE3(self, wu):  # [...,3]
        w, u = wu.split([3, 3], dim=-1)
        wx = self.skew_symmetric(w)
        theta = w.norm(dim=-1)[..., None, None]
        eye = torch.eye(3, device=w.device, dtype=torch.float32)
        A = self.taylor_A(theta)
        B = self.taylor_B(theta)
        C = self.taylor_C(theta)
        R = eye + A * wx + B * wx @ wx
        V = eye + B * wx + C * wx @ wx
        Rt = torch.cat([R, (V @ u[..., None])], dim=-1)
        return Rt

    def SE3_to_se3(self, Rt, eps=1e-8):  # [...,3,4]
        R, t = Rt.split([3, 1], dim=-1)
        w = self.SO3_to_so3(R)
        wx = self.skew_symmetric(w)
        theta = w.norm(dim=-1)[..., None, None]
        eye = torch.eye(3, device=w.device, dtype=torch.float32)
        A = self.taylor_A(theta)
        B = self.taylor_B(theta)
        invV = eye - 0.5 * wx + (1 - A / (2 * B)) / (theta**2 + eps) * wx @ wx
        u = (invV @ t)[..., 0]
        wu = torch.cat([w, u], dim=-1)
        return wu

    def skew_symmetric(self, w):
        w0, w1, w2 = w.unbind(dim=-1)
        zero = torch.zeros_like(w0)
        wx = torch.stack(
            [
                torch.stack([zero, -w2, w1], dim=-1),
                torch.stack([w2, zero, -w0], dim=-1),
                torch.stack([-w1, w0, zero], dim=-1),
            ],
            dim=-2,
        )
        return wx

    def taylor_A(self, x, nth=10):
        # Taylor expansion of sin(x)/x.
        ans = torch.zeros_like(x)
        denom = 1.0
        for i in range(nth + 1):
            if i > 0:
                denom *= (2 * i) * (2 * i + 1)
            ans = ans + (-1) ** i * x ** (2 * i) / denom
        return ans

    def taylor_B(self, x, nth=10):
        # Taylor expansion of (1-cos(x))/x**2.
        ans = torch.zeros_like(x)
        denom = 1.0
        for i in range(nth + 1):
            denom *= (2 * i + 1) * (2 * i + 2)
            ans = ans + (-1) ** i * x ** (2 * i) / denom
        return ans

    def taylor_C(self, x, nth=10):
        # Taylor expansion of (x-sin(x))/x**3.
        ans = torch.zeros_like(x)
        denom = 1.0
        for i in range(nth + 1):
            denom *= (2 * i + 2) * (2 * i + 3)
            ans = ans + (-1) ** i * x ** (2 * i) / denom
        return ans


class Quaternion:
    def q_to_R(self, q):  # [...,4]
        # https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
        qa, qb, qc, qd = q.unbind(dim=-1)
        R = torch.stack(
            [
                torch.stack([1 - 2 * (qc**2 + qd**2), 2 * (qb * qc - qa * qd), 2 * (qa * qc + qb * qd)], dim=-1),
                torch.stack([2 * (qb * qc + qa * qd), 1 - 2 * (qb**2 + qd**2), 2 * (qc * qd - qa * qb)], dim=-1),
                torch.stack([2 * (qb * qd - qa * qc), 2 * (qa * qb + qc * qd), 1 - 2 * (qb**2 + qc**2)], dim=-1),
            ],
            dim=-2,
        )
        return R

    def R_to_q(self, R, eps=1e-6):  # [...,3,3]
        # https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
        row0, row1, row2 = R.unbind(dim=-2)
        R00, R01, R02 = row0.unbind(dim=-1)
        R10, R11, R12 = row1.unbind(dim=-1)
        R20, R21, R22 = row2.unbind(dim=-1)
        t = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
        r = (1 + t + eps).sqrt()
        qa = 0.5 * r
        qb = (R21 - R12).sign() * 0.5 * (1 + R00 - R11 - R22 + eps).sqrt()
        qc = (R02 - R20).sign() * 0.5 * (1 - R00 + R11 - R22 + eps).sqrt()
        qd = (R10 - R01).sign() * 0.5 * (1 - R00 - R11 + R22 + eps).sqrt()
        q = torch.stack([qa, qb, qc, qd], dim=-1)
        return q

    def invert(self, q):  # [...,4]
        qa, qb, qc, qd = q.unbind(dim=-1)
        norm = q.norm(dim=-1, keepdim=True)
        q_inv = torch.stack([qa, -qb, -qc, -qd], dim=-1) / norm**2
        return q_inv

    def product(self, q1, q2):  # [...,4]
        q1a, q1b, q1c, q1d = q1.unbind(dim=-1)
        q2a, q2b, q2c, q2d = q2.unbind(dim=-1)
        hamil_prod = torch.stack(
            [
                q1a * q2a - q1b * q2b - q1c * q2c - q1d * q2d,
                q1a * q2b + q1b * q2a + q1c * q2d - q1d * q2c,
                q1a * q2c - q1b * q2d + q1c * q2a + q1d * q2b,
                q1a * q2d + q1b * q2c - q1c * q2b + q1d * q2a,
            ],
            dim=-1,
        )
        return hamil_prod

    def interpolate(self, q1, q2, alpha):  # [...,4],[...,4],[...,1]
        # https://en.wikipedia.org/wiki/Slerp
        cos_angle = (q1 * q2).sum(dim=-1, keepdim=True)  # [...,1]
        flip = cos_angle < 0
        q1 = q1 * (~flip) - q1 * flip  # [...,4]
        theta = cos_angle.abs().acos()  # [...,1]
        slerp = (((1 - alpha) * theta).sin() * q1 + (alpha * theta).sin() * q2) / theta.sin()  # [...,4]
        return slerp


pose = Pose()
lie = Lie()
quaternion = Quaternion()


def to_hom(X):
    # Get homogeneous coordinates of the input.
    X_hom = torch.cat([X, torch.ones_like(X[..., :1])], dim=-1)
    return X_hom


# Basic operations of transforming 3D points between world/camera/image coordinates.
def world2cam(X, pose):  # [B,N,3]
    X_hom = to_hom(X)
    return X_hom @ pose.transpose(-1, -2)


def cam2img(X, cam_intr):
    return X @ cam_intr.transpose(-1, -2)


def img2cam(X, cam_intr):
    _dtype = cam_intr.dtype
    X = X.float()
    cam_intr = cam_intr.float()
    result = X @ cam_intr.inverse().transpose(-1, -2)
    return result.to(dtype=_dtype)


def cam2world(X, pose):
    _dtype = pose.dtype
    X = X.float()
    pose = pose.float()
    X_hom = to_hom(X)
    pose_inv = Pose().invert(pose)
    result = X_hom @ pose_inv.transpose(-1, -2)
    return result.to(dtype=_dtype)


def angle_to_rotation_matrix(a, axis):
    # Get the rotation matrix from Euler angle around specific axis.
    roll = dict(X=1, Y=2, Z=0)[axis]
    if isinstance(a, float):
        a = torch.tensor(a)
    zero = torch.zeros_like(a)
    eye = torch.ones_like(a)
    M = torch.stack(
        [
            torch.stack([a.cos(), -a.sin(), zero], dim=-1),
            torch.stack([a.sin(), a.cos(), zero], dim=-1),
            torch.stack([zero, zero, eye], dim=-1),
        ],
        dim=-2,
    )
    M = M.roll((roll, roll), dims=(-2, -1))
    return M


def get_center_and_ray(pose, intr, image_size):
    """
    Args:
        pose (tensor [3,4]/[B,3,4]): Camera pose.
        intr (tensor [3,3]/[B,3,3]): Camera intrinsics.
        image_size (list of int): Image size.
    Returns:
        center_3D (tensor [HW,3]/[B,HW,3]): Center of the camera.
        ray (tensor [HW,3]/[B,HW,3]): Ray of the camera with depth=1 (note: not unit ray).
    """
    assert (
        pose.dtype == torch.float32 and intr.dtype == torch.float32
    ), f"pose and intr should be float32, got {pose.dtype} and {intr.dtype}"

    H, W = image_size
    # Given the intrinsic/extrinsic matrices, get the camera center and ray directions.
    with torch.no_grad():
        # Compute image coordinate grid.
        X, Y = get_pixel_grid(W, H, pose.device, normalized_coordinate=False)  # [H,W]
        xy_grid = torch.stack([X, Y], dim=-1).view(-1, 2)  # [HW,2]
    # Compute center and ray.
    if len(pose.shape) == 3:
        batch_size = len(pose)
        xy_grid = xy_grid.repeat(batch_size, 1, 1)  # [B,HW,2]
    grid_3D = img2cam(to_hom(xy_grid), intr)  # [HW,3]/[B,HW,3]
    center_3D = torch.zeros_like(grid_3D)  # [HW,3]/[B,HW,3]
    # Transform from camera to world coordinates.
    grid_3D = cam2world(grid_3D, pose)  # [HW,3]/[B,HW,3]
    center_3D = cam2world(center_3D, pose)  # [HW,3]/[B,HW,3]
    ray = grid_3D - center_3D  # [B,HW,3]
    return center_3D, ray


def get_pixel_grid(width: int, height: int, device: torch.device, normalized_coordinate: bool = False):
    """Generate pixel grid given the image size.

    Args:
        width (int): image width
        height (int): image height
        device (torch.device)
        normalized_coordinate (bool, optional): normalized coordinate is between 0 and 1. Defaults to False.

    Returns:
        torch.tensor: x,y pixel grid
    """
    y_range = torch.arange(height, dtype=torch.float32, device=device).add_(0.5)
    x_range = torch.arange(width, dtype=torch.float32, device=device).add_(0.5)
    if normalized_coordinate:
        y_range = y_range / height
        x_range = x_range / width
    y, x = torch.meshgrid(y_range, x_range, indexing="ij")  # [H, W]
    return x, y


def get_3D_points_from_dist(
    center: torch.tensor, ray_unit: torch.tensor, dist: torch.tensor, multiple_samples_per_ray: bool = False
):
    """Convert dist to 3D points in the world coordinate.

    Args:
        center (torch.tensor): camer center in world coordinates, [..., 3]
        ray_unit (torch.tensor): ray directions (unit vector), [..., 3]
        dist (torch.tensor): distance along the ray, [..., 1] or [..., N_samples, 1]
                             if sampling muliple points along rays
        multiple_samples_per_ray (bool): If True, dist is [..., N_samples, 1]

    Returns:
        torch.tensor: [..., 3] or [..., N_samples, 3]
    """
    assert torch.allclose(
        ray_unit.norm(dim=-1), torch.ones_like(ray_unit.norm(dim=-1))
    ), f"ray_unit norm is not equal to 1, max {ray_unit.norm(dim=-1).max()} min {ray_unit.norm(dim=-1).min()}"
    if multiple_samples_per_ray:
        assert len(dist.shape) == len(center.shape) + 1
        center, ray_unit = center[..., None, :], ray_unit[..., None, :]  # [...,1,3]
    else:
        assert len(dist.shape) == len(center.shape), f"dist shape {dist.shape} center shape {center.shape}"
    points_3D = center + ray_unit * dist  # [...,3]/[...,N_samples,3]
    return points_3D


def get_3D_points_from_depth(
    center: torch.tensor, ray: torch.tensor, depth: torch.tensor, multiple_samples_per_ray: bool = False
):
    """Convert depth to 3D points in the world coordinate.
    NOTE: this function assuems the ray is NOT noramlized and returned directly from get_center_and_ray()!!

    Args:
        center (torch.tensor): camer center in world coordinates, [..., 3]
        ray (torch.tensor): ray directions (z component is 1), [..., 3]
        depth (torch.tensor): z depth from camera center, [..., 1] or [..., N_samples, 1]
                              if sampling muliple points along rays
        multiple_samples_per_ray (bool): If True, depth is [..., N_samples, 1]

    Returns:
        torch.tensor: [..., 3] or [..., N_samples, 3]
    """
    if multiple_samples_per_ray:
        assert len(depth.shape) == len(center.shape) + 1
        center, ray = center[..., None, :], ray[..., None, :]  # [...,1,3]
    else:
        assert len(depth.shape) == len(center.shape)
    points_3D = center + ray * depth  # [...,3]/[...,N,3]
    return points_3D


def convert_NDC(center, ray, intr, near=1):
    # Shift camera center (ray origins) to near plane (z=1).
    # (Unlike conventional NDC, we assume the cameras are facing towards the +z direction.)
    center = center + (near - center[..., 2:]) / ray[..., 2:] * ray
    # Projection.
    cx, cy, cz = center.unbind(dim=-1)  # [...,R]
    rx, ry, rz = ray.unbind(dim=-1)  # [...,R]
    scale_x = intr[..., 0, 0] / intr[..., 0, 2]  # [...]
    scale_y = intr[..., 1, 1] / intr[..., 1, 2]  # [...]
    cnx = scale_x[..., None] * (cx / cz)
    cny = scale_y[..., None] * (cy / cz)
    cnz = 1 - 2 * near / cz
    rnx = scale_x[..., None] * (rx / rz - cx / cz)
    rny = scale_y[..., None] * (ry / rz - cy / cz)
    rnz = 2 * near / cz
    center_ndc = torch.stack([cnx, cny, cnz], dim=-1)  # [...,R,3]
    ray_ndc = torch.stack([rnx, rny, rnz], dim=-1)  # [...,R,3]
    return center_ndc, ray_ndc


def convert_NDC2(center, ray, intr):
    # Similar to convert_NDC() but shift the ray origins to its own image plane instead of the global near plane.
    # Also this version is much more interpretable.
    scale_x = intr[..., 0, 0] / intr[..., 0, 2]  # [...]
    scale_y = intr[..., 1, 1] / intr[..., 1, 2]  # [...]
    # Get the metric image plane (i.e. new "center"): (sx*cx/cz, sy*cy/cz, 1-2/cz).
    center = center + ray  # This is the key difference.
    cx, cy, cz = center.unbind(dim=-1)  # [...,R]
    image_plane = torch.stack([scale_x[..., None] * cx / cz, scale_x[..., None] * cy / cz, 1 - 2 / cz], dim=-1)
    # Get the infinity plane: (sx*rx/rz, sy*ry/rz, 1).
    rx, ry, rz = ray.unbind(dim=-1)  # [...,R]
    inf_plane = torch.stack([scale_x[..., None] * rx / rz, scale_y[..., None] * ry / rz, torch.ones_like(rz)], dim=-1)
    # The NDC ray is the difference between the two planes, assuming t \in [0,1].
    ndc_ray = inf_plane - image_plane
    return image_plane, ndc_ray


def rotation_distance(R1, R2, eps=1e-7):
    # http://www.boris-belousov.net/2016/12/01/quat-dist/
    R_diff = R1 @ R2.transpose(-2, -1)
    trace = R_diff[..., 0, 0] + R_diff[..., 1, 1] + R_diff[..., 2, 2]
    angle = ((trace - 1) / 2).clamp(-1 + eps, 1 - eps).acos_()  # numerical stability near -1/+1
    return angle


def get_oscil_novel_view_poses(N=60, angle=0.05, dist=5):
    # Create circular viewpoints (small oscillations).
    theta = torch.arange(N) / N * 2 * np.pi
    R_x = angle_to_rotation_matrix((theta.sin() * angle).asin(), "X")
    R_y = angle_to_rotation_matrix((theta.cos() * angle).asin(), "Y")
    pose_rot = pose(R=R_y @ R_x)
    pose_shift = pose(t=[0, 0, dist])
    pose_oscil = pose.compose([pose.invert(pose_shift), pose_rot, pose_shift])
    return pose_oscil


def cross_product_matrix(x):
    """Matrix form of cross product opertaion.

    param x: [3,] tensor.
    return: [3, 3] tensor representing the matrix form of cross product.
    """
    return torch.tensor(
        [
            [0, -x[2], x[1]],
            [x[2], 0, -x[0]],
            [
                -x[1],
                x[0],
                0,
            ],
        ]
    )


def essential_matrix(poses):
    """Compute Essential Matrix from a relative pose.

    param poses: [views, 3, 4] tensor representing relative poses.
    return: [views, 3, 3] tensor representing Essential Matrix.
    """
    r = poses[..., 0:3]
    t = poses[..., 3]
    tx = torch.stack([cross_product_matrix(tt) for tt in t], axis=0)
    return tx @ r


def fundamental_matrix(poses, intr1, intr2):
    """Compute Fundamental Matrix from a relative pose and intrinsics.

    param poses: [views, 3, 4] tensor representing relative poses.
          intr1: [3, 3] tensor. Camera intrinsic of reference image.
          intr2: [views, 3, 3] tensor. Camera Intrinsic of target image.
    return: [views, 3, 3] tensor representing Fundamental Matrix.
    """
    return intr2.inverse().transpose(-1, -2) @ essential_matrix(poses) @ intr1.inverse()


def get_ray_depth_plane_intersection(center, ray, depths):
    """Compute the intersection of a ray with a depth plane.
    Args:
        center (tensor [B,HW,3]): Camera center of the target pose.
        ray (tensor [B,HW,3]): Ray direction of the target pose.
        depth (tensor [L]): The depth values from the source view (e.g. for MPI planes).
    Returns:
        intsc_points (tensor [B,HW,L,3]): Intersecting 3D points with the MPI.
    """
    # Each 3D point x along the ray v from center c can be written as x = c+t*v.
    # Plane equation: n@x = d, where normal n = (0,0,1), d = depth.
    # --> t = (d-n@c)/(n@v).
    # --> x = c+t*v = c+(d-n@c)/(n@v)*v.
    center, ray = center[:, :, None], ray[:, :, None]  # [B,HW,L,3], [B,HW,1,3]
    depths = depths[None, None, :, None]  # [1,1,L,1]
    intsc_points = center + (depths - center[..., 2:]) / ray[..., 2:] * ray  # [B,HW,L,3]
    return intsc_points


def unit_view_vector_to_rotation_matrix(v, axes="ZYZ"):
    """
    Args:
        v (tensor [...,3]): Unit vectors on the view sphere.
        axes: rotation axis order.

    Returns:
        rotation_matrix (tensor [...,3,3]): rotation matrix R @ v + [0, 0, 1] = 0.
    """
    alpha = torch.arctan2(v[..., 1], v[..., 0])  # [...]
    beta = np.pi - v[..., 2].arccos()  # [...]
    euler_angles = torch.stack([torch.ones_like(alpha) * np.pi / 2, -beta, alpha], dim=-1)  # [...,3]
    rot2 = angle_to_rotation_matrix(euler_angles[..., 2], axes[2])  # [...,3,3]
    rot1 = angle_to_rotation_matrix(euler_angles[..., 1], axes[1])  # [...,3,3]
    rot0 = angle_to_rotation_matrix(euler_angles[..., 0], axes[0])  # [...,3,3]
    rot = rot2 @ rot1 @ rot0  # [...,3,3]
    return rot.transpose(-2, -1)


def sample_on_spherical_cap(anchor, N, max_angle, min_angle=0.0):
    """Sample n points on the view hemisphere within the angle to x.
    Args:
        anchor (tensor [...,3]): Reference 3-D unit vector on the view hemisphere.
        N (int): Number of sampled points.
        max_angle (float): Sampled points should have max angle to x.
    Returns:
        sampled_points (tensor [...,N,3]): Sampled points on the spherical caps.
    """
    batch_shape = anchor.shape[:-1]
    # First, sample uniformly on a unit 2D disk.
    radius = torch.rand(*batch_shape, N, device=anchor.device)  # [...,N]
    h_max = 1 - np.cos(max_angle)  # spherical cap height
    h_min = 1 - np.cos(min_angle)  # spherical cap height
    radius = (radius * (h_max - h_min) + h_min) / h_max
    theta = torch.rand(*batch_shape, N, device=anchor.device) * 2 * np.pi  # [...,N]
    x = radius.sqrt() * theta.cos()  # [...,N]
    y = radius.sqrt() * theta.sin()  # [...,N]
    # Reparametrize to a unit spherical cap with height h.
    # http://marc-b-reynolds.github.io/distribution/2016/11/28/Uniform.html
    k = h_max * radius  # [...,N]
    s = (h_max * (2 - k)).sqrt()  # [...,N]
    points = torch.stack([s * x, s * y, 1 - k], dim=-1)  # [...,N,3]
    # Transform to center around the anchor.
    ref_z = torch.tensor([0.0, 0.0, 1.0], device=anchor.device)
    v = -anchor.cross(ref_z)  # [...,3]
    ss_v = lie.skew_symmetric(v)  # [...,3,3]
    R = torch.eye(3, device=anchor.device) + ss_v + ss_v @ ss_v / (1 + anchor @ ref_z)[..., None, None]  # [...,3,3]
    points = points @ R.transpose(-2, -1)  # [...,N,3]
    return points


def sample_on_spherical_cap_northern(anchor, N, max_angle, away_from=None, max_reject_count=None):
    """Sample n points only the northern view hemisphere within the angle to x."""

    def find_invalid_points(points):
        southern = points[..., 2] < 0  # [...,N]
        if away_from is not None:
            cosine_ab = (away_from * anchor).sum(dim=-1, keepdim=True)  # [...,1]
            cosine_ac = (away_from[..., None, :] * points).sum(dim=-1)  # [...,N]
            not_outwards = cosine_ab < cosine_ac  # [...,N]
            invalid = southern | not_outwards
        else:
            invalid = southern
        return invalid

    assert (anchor[..., 2] > 0).all()
    assert anchor.norm(dim=-1).allclose(torch.ones_like(anchor[..., 0]))
    points = sample_on_spherical_cap(anchor, N, max_angle)  # [...,N,3]
    invalid = find_invalid_points(points)
    count = 0
    while invalid.any():
        # Reject and resample.
        points_resample = sample_on_spherical_cap(anchor, N, max_angle)
        points[invalid] = points_resample[invalid]
        invalid = find_invalid_points(points)
        count += 1
        if max_reject_count and count > max_reject_count:
            points = anchor.repeat(N, 1)
    return points


def depth_to_pointcloud(depth: torch.tensor, intr: torch.tensor, extr: torch.tensor):
    """Convert depth to pointcloud.
    Args:
        depth (torch.tensor): [1,H,W]/[B,1,H,W]
        intr (torch.tensor): [3,3]/[B,3,3]
        extr (torch.tensor): [3,4]/[B,3,4]

    Returns:
        pc (torch.tensor): [HW,3]
    """

    assert (
        len(depth.shape) == len(intr.shape) + 1
    ), f"dist ({depth.shape}) and intr ({intr.shape}) should have the same batch size"
    # convert depth to pointcloud
    center, ray = get_center_and_ray(extr, intr, depth.shape[-2:])
    depth = depth.view(*center.shape[:-1], 1)  # [HW, 1]/[B,HW,1]
    pc = get_3D_points_from_depth(center, ray, depth)
    return pc  # HW,3/B,HW,3
