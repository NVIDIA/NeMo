#
from typing import Optional, Union
import numpy as np



import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_dct as dct


from einops.layers.torch import Rearrange
from einops import rearrange


from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
    Qwen2VLImageProcessor as _Base,
)
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

from transformers.image_processing_utils import BatchFeature
from transformers.image_utils import (
    ChannelDimension, PILImageResampling, ImageInput,VideoInput,
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    get_image_size,
    infer_channel_dimension_format,
    is_scaled_image,
    make_flat_list_of_images,
    make_list_of_images,
    to_numpy_array,
    valid_images,
    validate_preprocess_arguments,
    )

from transformers.image_transforms import(
    convert_to_rgb,
    resize,
    to_channel_dimension_format, ) 

from transformers.utils import TensorType, logging

logger = logging.get_logger(__name__)

def patchfication(x):

    return rearrange(x, 'b c (h ph) (w pw) -> b c ph pw h w', ph=2, pw=2)

def patchfication_nonbatch(x):
    return rearrange(x, 'c (h ph) (w pw) -> c ph pw h w', ph=2, pw=2)

def rea(x):
    return rearrange(x, 'c h w ph pw -> c (h ph) (w pw)')
def b_rea(x):
    return rearrange(x, 'b c h w ph pw -> b c (h ph) (w pw)')

def rea_f(x):
    return  rearrange(x, 'c (h ph) (w pw) -> c h w ph pw', ph=4, pw=4)
def b_rea_f(x):
    return  rearrange(x, 'b c (h ph) (w pw) -> b c h w ph pw', ph=4, pw=4)


@torch.cuda.nvtx.range('process_channel')
def process_channel(channel, qmap):
    """
    Dequantize and apply IDCT for each 8x8 block.
    """
    if len(channel.shape) == 5:
        channel = channel[:,:,:,:,:] * qmap[None,None,None,:,:]#
        channel = dct.idct_2d(channel, norm='ortho')
        channel =  rea(channel)
    elif len(channel.shape) == 6:
        channel = channel[:,:,:,:,:,:] * qmap[:,None,None,None,:,:]
        channel = dct.idct_2d(channel, norm='ortho')
        channel =  b_rea(channel)
    return channel


@torch.cuda.nvtx.range('split_rgb_to_ycbcr')
def split_rgb_to_ycbcr(rgb):
    """
    Convert RGB to YCbCr channels using OpenCV's RGB to YCbCr conversion.
    """
    rgb = rgb * 255

    if len(rgb.shape) == 3:
        r, g, b = rgb[0], rgb[1], rgb[2]
    elif len(rgb.shape) == 4:
        r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    else:
        raise ValueError("Input tensor should have 3 or 4 dimensions")

    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
    try:
        return y.unsqueeze(-3) , cb.unsqueeze(-3) , cr.unsqueeze(-3)
    except:
        return np.expand_dims(y,axis=-3) , np.expand_dims(cb,axis=-3) ,np.expand_dims(cr,axis=-3)




@torch.cuda.nvtx.range('process_channel_forward')
def process_channel_forward(channel):
    """
    Dequantize and apply IDCT for each 8x8 block.
    """
    if len(channel.shape) == 3:
        channel = rea_f(channel)
        channel = dct.dct_2d(channel,norm='ortho')#*np.sqrt(2.0)
    elif len(channel.shape) == 4:
        channel = b_rea_f(channel)
        channel = dct.dct_2d(channel,norm='ortho')#*(32)#*np.sqrt(2.0)
    return channel

def _is_torch(x): return isinstance(x, torch.Tensor)

def _resize_2d(x, out_h, out_w, mode="nearest"):
    """2D/3D/4D -> 원하는 (H,W) 로 리사이즈. torch/np 모두 지원."""
    if _is_torch(x):
        if x.ndim == 2:          # (H,W)
            x4 = x[None, None]   # (1,1,H,W)
            y4 = F.interpolate(x4, size=(out_h, out_w), mode=mode)
            return y4[0, 0]
        elif x.ndim == 3:        # (C,H,W)
            x4 = x[None]         # (1,C,H,W)
            y4 = F.interpolate(x4, size=(out_h, out_w), mode=mode)
            return y4[0]
        elif x.ndim == 4:        # (N,C,H,W)
            return F.interpolate(x, size=(out_h, out_w), mode=mode)
        else:
            raise ValueError(f"_resize_2d: unsupported torch ndim={x.ndim}")
    else:
        # numpy: 최근접 리사이즈(정수배 가정) → stride 샘플링
        in_h, in_w = x.shape[-2], x.shape[-1]
        sh = int(round(in_h / out_h)); sw = int(round(in_w / out_w))
        if sh <= 0 or sw <= 0 or in_h % sh or in_w % sw:
            raise ValueError(f"_resize_2d numpy: invalid scale {(in_h,in_w)}->{(out_h,out_w)}")
        if x.ndim == 2:          # (H,W)
            return x[::sh, ::sw]
        elif x.ndim == 3:        # (C,H,W)
            return x[:, ::sh, ::sw]
        elif x.ndim == 4:        # (N,C,H,W)
            return x[:, :, ::sh, ::sw]
        else:
            raise ValueError(f"_resize_2d: unsupported numpy ndim={x.ndim}")


def _cat(xs, axis):
    return torch.cat(xs, dim=axis) if _is_torch(xs[0]) else np.concatenate(xs, axis=axis)

def _permute(x, order):
    return x.permute(*order) if _is_torch(x) else np.transpose(x, axes=order)

def _reshape(x, shape):
    return x.reshape(*shape) if _is_torch(x) else np.reshape(x, shape)

def _to_float(x):
    if _is_torch(x):
        return x if x.dtype.is_floating_point else x.float()
    else:
        return x if np.issubdtype(x.dtype, np.floating) else x.astype(np.float32)



def rgb_to_spec_tokenize(img, mode=2):
    """
    img: torch.Tensor or np.ndarray
         shape ...xH x W  (마지막 두 축이 H,W)
    mode: 0=4:4:4, 1=4:2:2 (W/2), 2=4:2:0 (H/2,W/2)
    """
    h, w = img.shape[-2], img.shape[-1]
    y, cb, cr = split_rgb_to_ycbcr(img)  # 타입 유지 가정

    # --- chroma subsampling ---
    if mode == 0:
        # 4:4:4 (그대로)
        pass
    elif mode == 1:
        # 4:2:2  → (H, W/2)
            cb = _resize_2d(cb, h, w // 2, mode="nearest")
            cr = _resize_2d(cr, h, w // 2, mode="nearest")
    elif mode == 2:
        # 4:2:0  → (H/2, W/2)
            cb = _resize_2d(cb, h // 2, w // 2, mode="nearest")
            cr = _resize_2d(cr, h // 2, w // 2, mode="nearest")
    else:
        raise ValueError(f"Unknown mode={mode}")

    # --- channel processing ---
    # 형 일치 + float 보장 후 -128
    y  = process_channel_forward(_to_float(y)  - 128)
    cb = process_channel_forward(_to_float(cb) - 128)
    cr = process_channel_forward(_to_float(cr) - 128)

    if y.ndim == 5:
        # cbcr 쌓기 (C 채널 기준)
        cbcr = _cat([cb, cr], axis=0)  # (2C, H', W')

        # 아래 permute/reshape는 원래 코드의 형태를 그대로 보존
        # y: (C,H,W) -> (C, ph, pw, nH, nW) 순을 (C,nH,nW,ph,pw)로 바꾼 뒤 패치 플랫
        y_ = _permute(y, (0, 3, 4, 1, 2)) if y.ndim == 5 else y  # 안전 처리(이미 patchified일 수 있으니)
        y  = patchfication_nonbatch(_reshape(y_, (-1, h // 4, w // 4))).reshape(-1, h // 8, w // 8)

        cbcr_ = _permute(cbcr, (0, 3, 4, 1, 2)) if cbcr.ndim == 5 else cbcr
        cbcr  = _reshape(cbcr_, (-1, h // 8, w // 8))

        return _cat([y, cbcr], axis=0)

    elif y.ndim == 6:
        # (N,C,H,W) 케이스
        cbcr = _cat([cb, cr], axis=1)  # (N, 2C, H', W')

        y_ = _permute(y, (0, 1, 4, 5, 2, 3)) if y.ndim == 6 else y
        y  = patchfication(
                _reshape(y_, (y.shape[0], -1, h // 4, w // 4))
             ).reshape(y.shape[0], -1, h // 8, w // 8)

        cbcr_ = _permute(cbcr, (0, 1, 4, 5, 2, 3)) if cbcr.ndim == 6 else cbcr
        cbcr  = _reshape(cbcr_, (y.shape[0], -1, h // 8, w // 8))

        return _cat([y, cbcr], axis=1)

    # fallback (혹시 위에서 return 안 됐다면)
    return y, _cat([cb, cr], axis=1 if (y.ndim == 4) else 0)



class Qwen2VLImageProcessorDCT(_Base):
    def _preprocess(
        self,
        images: Union[ImageInput, VideoInput],
        do_resize: Optional[bool] = None,
        size: Optional[dict[str, int]] = None,
        resample: PILImageResampling = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, list[float]]] = None,
        image_std: Optional[Union[float, list[float]]] = None,
        patch_size: Optional[int] = None,
        temporal_patch_size: Optional[int] = None,
        merge_size: Optional[int] = None,
        do_convert_rgb: Optional[bool] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        """
        Preprocess an image or batch of images. Copy of the `preprocess` method from `CLIPImageProcessor`.

        Args:
            images (`ImageInput`):
                Image or batch of images to preprocess. Expects pixel values ranging from 0 to 255. If pixel values range from 0 to 1, set `do_rescale=False`.
            vision_info (`list[Dict]`, *optional*):
                Optional list of dictionaries containing additional information about vision inputs.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            size (`dict[str, int]`, *optional*, defaults to `self.size`):
                Size of the image after resizing. `shortest_edge` and `longest_edge` keys must be present.
            resample (`PILImageResampling`, *optional*, defaults to `self.resample`):
                Resampling filter to use if resizing the image. This can be one of the `PILImageResampling` enums.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the image.
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                Scale factor to use if rescaling the image.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image.
            image_mean (`float` or `list[float]`, *optional*, defaults to `self.image_mean`):
                Mean to use if normalizing the image. Can be a float or a list of floats corresponding to the number of channels in the image.
            image_std (`float` or `list[float]`, *optional*, defaults to `self.image_std`):
                Standard deviation to use if normalizing the image. Can be a float or a list of floats corresponding to the number of channels in the image.
            patch_size (`int`, *optional*, defaults to `self.patch_size`):
                The spatial patch size of the vision encoder.
            temporal_patch_size (`int`, *optional*, defaults to `self.temporal_patch_size`):
                The temporal patch size of the vision encoder.
            merge_size (`int`, *optional*, defaults to `self.merge_size`):
                The merge size of the vision encoder to llm encoder.
            do_convert_rgb (`bool`, *optional*, defaults to `self.do_convert_rgb`):
                Whether to convert the image to RGB.
            data_format (`ChannelDimension`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format for the output image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - Unset: Use the channel dimension format of the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.   - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
        """
        images = make_list_of_images(images)

        if do_convert_rgb:
            images = [convert_to_rgb(image) for image in images]

        # All transformations expect numpy arrays.
        images = [to_numpy_array(image) for image in images]

        if do_rescale and is_scaled_image(images[0]):
            logger.warning_once(
                "It looks like you are trying to rescale already rescaled images. If the input"
                " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
            )
        if input_data_format is None:
            # We assume that all images have the same channel dimension format.
            input_data_format = infer_channel_dimension_format(images[0])

        height, width = get_image_size(images[0], channel_dim=input_data_format)
        resized_height, resized_width = height, width
        processed_images = []
        for image in images:
            if do_resize:
                resized_height, resized_width = smart_resize(
                    height,
                    width,
                    factor=patch_size * merge_size,
                    min_pixels=size["shortest_edge"],
                    max_pixels=size["longest_edge"],
                )
                image = resize(
                    image, size=(resized_height, resized_width), resample=resample, input_data_format=input_data_format
                )

            if do_rescale:
                image = self.rescale(image, scale=rescale_factor, input_data_format=input_data_format)

            # if do_normalize:
            #     image = self.normalize(
            #         image=image, mean=image_mean, std=image_std, input_data_format=input_data_format
            #     )

            image = to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)
            processed_images.append(image)

        patches = np.array(processed_images)
        patches = rgb_to_spec_tokenize(torch.Tensor(patches)).numpy()
        resized_height, resized_width = resized_height//8, resized_width//8
        # patches = rgb_to_spec_tokenize((patches))#.numpy()

        # print(patches)
        print(patches.shape)
        print(patch_size)
        patch_size
        if data_format == ChannelDimension.LAST:
            patches = patches.transpose(0, 3, 1, 2)
        if patches.shape[0] % temporal_patch_size != 0:
            repeats = np.repeat(
                patches[-1][np.newaxis], temporal_patch_size - (patches.shape[0] % temporal_patch_size), axis=0
            )
            patches = np.concatenate([patches, repeats], axis=0)
        channel = patches.shape[1]
        grid_t = patches.shape[0] // temporal_patch_size
        grid_h, grid_w = resized_height // patch_size, resized_width // patch_size
        # print(patches.shape)
        patches = patches.reshape(
            grid_t,
            temporal_patch_size,
            channel,
            grid_h // merge_size,
            merge_size,
            patch_size,
            grid_w // merge_size,
            merge_size,
            patch_size,
        )
        print(patches.shape)
        patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
        flatten_patches = patches.reshape(
            grid_t * grid_h * grid_w, channel * temporal_patch_size * patch_size * patch_size
        )

        return flatten_patches, (grid_t, grid_h, grid_w)
