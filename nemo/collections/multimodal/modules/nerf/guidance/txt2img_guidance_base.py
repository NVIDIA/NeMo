import torch.nn as nn


class Txt2ImgGuidanceBase(nn.Module):
    def __init__(self):
        super().__init__()
