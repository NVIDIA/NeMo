# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
import collections
import os

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from scipy import linalg
from TFinception_V3 import InceptionV3, SwAV, TFInceptionV3, Vgg16
from torch import nn


def network_init(network='inception'):
    # inception = inception_v3(pretrained=True, transform_input=False)
    # inception = inception.to('cuda')
    # inception.eval()
    # inception.fc = torch.nn.Sequential()

    if dist.is_initialized() and not is_local_master():
        # Make sure only the first process in distributed training downloads
        # the model, and the others will use the cache
        # noinspection PyUnresolvedReferences
        torch.distributed.barrier()

    if network == 'tf_inception':
        model = TFInceptionV3()
    elif network == 'inception':
        model = InceptionV3()
    elif network == 'vgg16':
        model = Vgg16()
    elif network == 'swav':
        model = SwAV()
    elif network == 'clean_inception':
        model = CleanInceptionV3()
    else:
        raise NotImplementedError(f'Network "{network}" is not supported!')

    if dist.is_initialized() and is_local_master():
        # Make sure only the first process in distributed training downloads
        # the model, and the others will use the cache
        # noinspection PyUnresolvedReferences
        dist.barrier()

    model = model.to('cuda').eval()
    return model


def _calculate_frechet_distance(act_1, act_2, eps=1e-6):
    mu1 = np.mean(act_1.cpu().numpy(), axis=0)
    sigma1 = np.cov(act_1.cpu().numpy(), rowvar=False)
    mu2 = np.mean(act_2.cpu().numpy(), axis=0)
    sigma2 = np.cov(act_2.cpu().numpy(), rowvar=False)
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    assert mu1.shape == mu2.shape, 'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, 'Training and test covariances have different dimensions'
    diff = mu1 - mu2
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; ' 'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            print('Imaginary component {}'.format(m))
            # raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    return {"FID": (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)}


def is_master():
    r"""check if current process is the master"""
    return get_rank() == 0


def get_rank():
    r"""Get rank of the thread."""
    rank = 0
    if dist.is_available():
        if dist.is_initialized():
            rank = dist.get_rank()
    return rank


def is_local_master():
    return torch.cuda.current_device() == 0


def load_or_compute_activations(
    act_path,
    data_loader,
    key_real,
    key_fake,
    generator=None,
    sample_size=None,
    preprocess=None,
    is_video=False,
    few_shot_video=False,
    network='inception',
    **kwargs,
):
    r"""Load mean and covariance from saved npy file if exists. Otherwise, compute the mean and covariance.

    Args:
        act_path (str or None): Location for the numpy file to store or to load the activations.
        data_loader (obj): PyTorch dataloader object.
        key_real (str): Dictionary key value for the real data.
        key_fake (str): Dictionary key value for the fake data.
        generator (obj): PyTorch trainer network.
        sample_size (int): How many samples to be used for computing the KID.
        preprocess (func): The preprocess function to be applied to the data.
        is_video (bool): Whether we are handling video sequences.
        few_shot_video (bool): If ``True``, uses few-shot video synthesis.
        network (str): Which recognition backbone to use.
    Returns:
        (torch.Tensor) Feature activations.
    """
    if act_path is not None and os.path.exists(act_path):
        # Loading precomputed activations.
        print('Load activations from {}'.format(act_path))
        act = torch.load(act_path, map_location='cpu').cuda()
    else:
        # Compute activations.
        if is_video:
            act = get_video_activations(
                data_loader, key_real, key_fake, generator, sample_size, preprocess, few_shot_video, network, **kwargs
            )
        else:
            act = get_activations(
                data_loader, key_real, key_fake, generator, sample_size, preprocess, True, network, **kwargs
            )
        if act_path is not None and is_local_master():
            print('Save activations to {}'.format(act_path))
            if not os.path.exists(os.path.dirname(act_path)):
                os.makedirs(os.path.dirname(act_path), exist_ok=True)
            torch.save(act, act_path)
    return act


@torch.no_grad()
def compute_fid(
    fid_path,
    data_loader,
    net_G,
    key_real='images',
    key_fake='fake_images',
    sample_size=None,
    preprocess=None,
    return_act=False,
    is_video=False,
    few_shot_video=False,
    **kwargs,
):
    r"""Compute the fid score.

    Args:
        fid_path (str): Location for the numpy file to store or to load the statistics.
        data_loader (obj): PyTorch dataloader object.
        net_G (obj): For image generation modes, net_G is the generator network.
            For video generation models, net_G is the trainer.
        key_real (str): Dictionary key value for the real data.
        key_fake (str): Dictionary key value for the fake data.
        sample_size (int or tuple): How many samples to be used.
        preprocess (func): The preprocess function to be applied to the data.
        return_act (bool): If ``True``, also returns feature activations of
            real and fake data.
        is_video (bool): Whether we are handling video sequences.
        few_shot_video (bool): If ``True``, uses few-shot video synthesis.
    Returns:
        (float): FID value.
    """
    print('Computing FID.')
    act_path = os.path.join(os.path.dirname(fid_path), 'activations_real.npy')
    # Get the fake mean and covariance.
    fake_act = load_or_compute_activations(
        None,
        data_loader,
        key_real,
        key_fake,
        net_G,
        sample_size,
        preprocess,
        is_video=is_video,
        few_shot_video=few_shot_video,
        **kwargs,
    )

    # Get the ground truth mean and covariance.
    real_act = load_or_compute_activations(
        act_path,
        data_loader,
        key_real,
        key_fake,
        None,
        sample_size,
        preprocess,
        is_video=is_video,
        few_shot_video=few_shot_video,
        **kwargs,
    )

    if is_master():
        fid = _calculate_frechet_distance(fake_act, real_act)["FID"]
        if return_act:
            return fid, real_act, fake_act
        else:
            return fid
    elif return_act:
        return None, None, None
    else:
        return None


def get_world_size():
    r"""Get world size. How many GPUs are available in this job."""
    world_size = 1
    if dist.is_available():
        if dist.is_initialized():
            world_size = dist.get_world_size()
    return world_size


def dist_all_gather_tensor(tensor):
    r""" gather to all ranks """
    world_size = get_world_size()
    if world_size < 2:
        return [tensor]
    tensor_list = [torch.ones_like(tensor) for _ in range(dist.get_world_size())]
    with torch.no_grad():
        dist.all_gather(tensor_list, tensor)
    return tensor_list


def to_device(data, device):
    r"""Move all tensors inside data to device.

    Args:
        data (dict, list, or tensor): Input data.
        device (str): 'cpu' or 'cuda'.
    """
    assert device in ['cpu', 'cuda']
    string_classes = (str, bytes)
    if isinstance(data, torch.Tensor):
        data = data.to(torch.device(device))
        return data
    elif isinstance(data, collections.abc.Mapping):
        return type(data)({key: to_device(data[key], device) for key in data})
    elif isinstance(data, collections.abc.Sequence) and not isinstance(data, string_classes):
        return type(data)([to_device(d, device) for d in data])
    else:
        return data


def to_cuda(data):
    r"""Move all tensors inside data to gpu.

    Args:
        data (dict, list, or tensor): Input data.
    """
    return to_device(data, 'cuda')


@torch.no_grad()
def get_activations(
    data_loader,
    key_real,
    key_fake,
    generator=None,
    sample_size=None,
    preprocess=None,
    align_corners=True,
    network='inception',
    **kwargs,
):
    r"""Compute activation values and pack them in a list.

    Args:
        data_loader (obj): PyTorch dataloader object.
        key_real (str): Dictionary key value for the real data.
        key_fake (str): Dictionary key value for the fake data.
        generator (obj): PyTorch trainer network.
        sample_size (int): How many samples to use for FID.
        preprocess (func): Pre-processing function to use.
        align_corners (bool): The ``'align_corners'`` parameter to be used for `torch.nn.functional.interpolate`.
        network (str): Which recognition backbone to use.
    Returns:
        batch_y (tensor): Inception features of the current batch. Note that only the master gpu will get it.
    """
    model = network_init(network)
    batch_y = []
    world_size = get_world_size()

    # Iterate through the dataset to compute the activation.
    for it, data in enumerate(data_loader):
        data = to_cuda(data)
        # Preprocess the data.
        if preprocess is not None:
            data = preprocess(data)
        # Load real data if the generator is not specified.
        if generator is None:
            images = data[key_real]
            if torch.max(images) > 1:
                images = images / 255.0  # convert RGB to (0,1)
        else:
            # Compute the generated image.
            text = data[1]['caption']  ### input is captions
            net_G_output = generator(text, **kwargs)
            images = net_G_output
        # Clamp the image for models that do not set the output to between
        # -1, 1. For models that employ tanh, this has no effect.
        images.clamp_(-1, 1)
        y = model(images, align_corners=align_corners)
        # y = network_forward(model, images, align_corners=align_corners)
        batch_y.append(y)
        if sample_size is not None and data_loader.batch_size * world_size * (it + 1) >= sample_size:
            # Reach the number of samples we need.
            break

    batch_y = torch.cat(dist_all_gather_tensor(torch.cat(batch_y)))
    if sample_size is not None:
        batch_y = batch_y[:sample_size]
    print(f"Computed feature activations of size {batch_y.shape}")
    return batch_y


@torch.no_grad()
def compute_fid_data(
    folder_to_store_real_act,
    data_loader_a,
    data_loader_b,
    key_a='images',
    key_b='images',
    sample_size=None,
    is_video=False,
    few_shot_video=False,
    network='inception',
    **kwargs,
):
    r"""Compute the fid score between two datasets.

    Args:
        folder_to_store_real_act (str): Location to store the statistics or to load the statistics.
        data_loader_a (obj): PyTorch dataloader object for dataset a.
        data_loader_b (obj): PyTorch dataloader object for dataset b.
        key_a (str): Dictionary key value for images in the dataset a.
        key_b (str): Dictionary key value for images in the dataset b.
        sample_size (int or None): How many samples to be used for computing the FID.
        is_video (bool): Whether we are handling video sequences.
        few_shot_video (bool): If ``True``, uses few-shot video synthesis.
        network (str): Which recognition backbone to use.
    Returns:
        (float): FID value.
    """
    print('Computing FID.')
    if folder_to_store_real_act is None:
        path_a = None
    else:
        path_a = os.path.join(os.path.dirname(folder_to_store_real_act), 'activations_a.npy')
    # min_data_size = min(len(data_loader_a.dataset), len(data_loader_b.dataset))
    # sample_size = min_data_size if sample_size is None else min(sample_size, min_data_size)

    act_a = load_or_compute_activations(
        path_a,
        data_loader_a,
        key_a,
        key_b,
        None,
        sample_size=sample_size,
        is_video=is_video,
        few_shot_video=few_shot_video,
        network=network,
        **kwargs,
    )
    act_b = load_or_compute_activations(
        None,
        data_loader_b,
        key_a,
        key_b,
        None,
        sample_size=sample_size,
        is_video=is_video,
        few_shot_video=few_shot_video,
        network=network,
        **kwargs,
    )
    print(act_a.shape, act_b.shape)
    if is_master():
        return _calculate_frechet_distance(act_a, act_b)["FID"]
