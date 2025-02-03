from nemo.collections.physicalai.tokenizer.models.cosmos_tokenizer.networks.configs import *
from nemo.collections.physicalai.tokenizer.models.cosmos_tokenizer.networks.continuous_video import CausalContinuousVideoTokenizer
from nemo.collections.physicalai.tokenizer.models.cosmos_tokenizer.networks.discrete_video import CausalDiscreteVideoTokenizer
from nemo.collections.physicalai.tokenizer.models.cosmos_tokenizer.video_cli import CausalVideoTokenizer
from nemo.collections.physicalai.tokenizer.models.cosmos_tokenizer.utils import load_model

import cv2
import numpy as np
import torch
import os

import importlib
import cosmos_tokenizer.video_lib
import mediapy as media

#####this will be needed in the training
model_cls = CausalContinuousVideoTokenizer(**continuous_video)
print("#"*20)
model_cls = torch.jit.load("/home/yihuih/coreai/cache/huggingface/hub/models--nvidia--Cosmos-1.0-Tokenizer-CV8x8x8/snapshots/01f87fd67cebc32f1a2fd9e99d4e9614a6b3743b/autoencoder.jit",
                        map_location="cpu")
print(model_cls)

tokenizer = CausalVideoTokenizer()
tokenizer.load_state_dict(model_cls.state_dict())

device = torch.device("cuda")
tokenizer = tokenizer.to(device=device, dtype=torch.bfloat16)
tokenizer.eval()



######this will be needed in the inference
tokenizer = CausalVideoTokenizer()
tokenizer._full_model = model_cls

device = torch.device("cuda")
tokenizer = tokenizer.to(device=device, dtype=torch.bfloat16)
tokenizer.eval()


#########################################
input_filepath = "/mnt/data/datasets/panda70m-test/6oQHGy6Dr7M.mp4"

# 3) Read the video from disk (shape = T x H x W x 3 in BGR).
input_video = media.read_video(input_filepath)[..., :3]
assert input_video.ndim == 4 and input_video.shape[-1] == 3, "Frames must have shape T x H x W x 3"

original_fps = 30
target_fps = 30

# Calculate the stride: how many frames to skip each time
stride = original_fps // target_fps  # e.g. 30 // 10 = 3

input_video = input_video[::stride]
input_video = input_video[:150]

sampled_input_filepath = "sampled.mp4"
media.write_video(sampled_input_filepath, input_video, fps=target_fps)


# 4) Expand dimensions to B x Tx H x W x C, since the CausalVideoTokenizer expects a batch dimension
#    in the input. (Batch size = 1 in this example.)
batched_input_video = np.expand_dims(input_video, axis=0)
temporal_window = 65

# 6) Use the tokenizer to autoencode (encode & decode) the video.
#    The output is a NumPy array with shape = B x T x H x W x C, range [0..255].
batched_output_video = tokenizer(batched_input_video,
                                 temporal_window=temporal_window)

# 7) Extract the single video from the batch (index 0).
output_video = batched_output_video[0]

print(input_video.shape, output_video.shape)
# 9) Save the reconstructed video to disk.
input_dir, input_filename = os.path.split(input_filepath)
filename, ext = os.path.splitext(input_filename)
output_filepath = f"reconstruct.mp4"
media.write_video(output_filepath, output_video, fps=target_fps)
print("Input video read from:\t", f"{os.getcwd()}/{input_filepath}")
print("Reconstruction saved:\t", f"{os.getcwd()}/{output_filepath}")

# 10) Visualization of the input video (left) and the reconstruction (right).
media.show_videos([input_video, output_video], ["Input Video", "Reconstructed Video"], height=720)
