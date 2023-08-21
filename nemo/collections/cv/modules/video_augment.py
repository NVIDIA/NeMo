from nemo.core.classes.module import NeuralModule
import torchaudio
import torchvision
from torch import nn
import torch
import random

class VideoAugmentation(NeuralModule):

    """ Video Augmentation for batched video input: input_signal shape (B, C, T, H, W) """

    def __init__(self, random_crop,  crop_size, horizontal_flip, time_masking, num_mask_second=1.0, spatial_masking=False, mean_frame=True):
        super().__init__()

        # Params
        self.random_crop = random_crop
        self.crop_size = crop_size
        self.horizontal_flip = horizontal_flip
        self.time_masking = time_masking
        self.spatial_masking = spatial_masking

        self.training_augments = nn.ModuleList()
        self.inference_augments = nn.ModuleList()

        # Random Crop
        if self.random_crop:
            self.training_augments.append(torchvision.transforms.RandomCrop(self.crop_size))
            self.inference_augments.append(torchvision.transforms.CenterCrop(self.crop_size))

        # Horizontal Flip
        if self.horizontal_flip:
            self.training_augments.append(torchvision.transforms.RandomHorizontalFlip())

        # Time Masking
        if self.time_masking:
            self.training_augments.append(VideoFrameMasking(num_mask_second=num_mask_second, mean_frame=mean_frame))

        # Spatial Masking
        if self.spatial_masking:
            self.training_augments.append(SpatialVideoMasking(mean_frame=mean_frame))

    @torch.no_grad()
    def forward(self, input_signal, length):

        if self.training:
            augments = self.training_augments
        else:
            augments = self.inference_augments

        for augment in augments:
            if isinstance(augment, VideoFrameMasking) or isinstance(augment, SpatialVideoMasking):
                input_signal = augment(input_signal, length)
            else:
                input_signal = augment(input_signal)

        return input_signal

class SpatialVideoMasking(NeuralModule):

    """ Spatial Video Mask

    Will mask videos frames in the spatial dimensions using horizontal and vertical masks

    params: 
        num_horizontal_masks: number of horizontal masks
        num_vertical_masks: number of vertical masks
        max_h: maximum width of horizontal mask  
        max_v: maximum width of vertical mask
        mean_frame: mask using video mean instead of zeros
    
    """

    def __init__(self, num_horizontal_masks=1, num_vertical_masks=1, max_h=30, max_v=30, mean_frame=True):
        super().__init__()

        self.num_horizontal_masks = num_horizontal_masks
        self.num_vertical_masks = num_vertical_masks
        self.max_h = max_h
        self.max_v = max_v
        self.mean_frame = mean_frame
        self.random = random.Random()

    def forward(self, input_signal, length):

        # (B, C, T, H, W)
        shape = input_signal.shape

        # Batch loop
        for b in range(shape[0]):

            # Mask Value
            mask_value = input_signal[b, :, :length[b]].mean() if self.mean_frame else 0.0

            # Horizontal Mask loop
            for i in range(self.num_horizontal_masks):

                # Start index
                x = self.random.randint(0, shape[3] - self.max_h)

                # Mask width
                w = self.random.randint(0, self.max_h)

                # Apply mask
                input_signal[b, :, :, x : x + w] = mask_value

            # Vertical Mask loop
            for i in range(self.num_vertical_masks):

                # Start index
                x = self.random.randint(0, shape[4] - self.max_v)

                # Mask width
                w = self.random.randint(0, self.max_v)

                # Apply mask
                input_signal[b, :, :, :, x : x + w] = mask_value

        return input_signal

class VideoFrameMasking(NeuralModule):

    """ Video Frame Mask:

    As explained in:
    "Visual Speech Recognition for Multiple Languages in the Wild"
    https://arxiv.org/abs/2202.13084

    S6 Time Masking
    We mask n consecutive frames with the mean frame of the video. 
    The duration tn is chosen from 0 to an upper bound nmax using a uniform distribution. 
    Since there is a large variance in the video lengths of the LRS2 and LRS3 datasets, we set the number of masks proportional to the sequence length. 
    Specifically, we use one mask per second, and for each mask, the maximum duration nmax is set to 0.4 seconds.

    """

    def __init__(self, T_second=0.4, num_mask_second=1.0, fps=25.0, mean_frame=True):
        super().__init__()

        self.T = int(T_second * fps)
        self.num_mask_second = num_mask_second
        self.mean_frame = mean_frame
        self.fps = fps
        self.random = random.Random()

    def forward(self, input_signal, length):

        # (B, C, T, H, W)
        shape = input_signal.shape

        # Batch loop
        for b in range(shape[0]):

            # Mask per second
            mT = int(length[b] / self.fps * self.num_mask_second)

            # Mask Value
            mask_value = input_signal[b, :, :length[b]].mean() if self.mean_frame else 0.0

            # Mask loop
            for i in range(mT):

                # Start left Frame
                x_left = self.random.randint(0, length[b] - self.T)

                # Mask width
                w = self.random.randint(0, self.T)

                # Apply mask
                input_signal[b, :, x_left : x_left + w] = mask_value

        return input_signal