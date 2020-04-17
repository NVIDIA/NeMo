# *****************************************************************************
#  Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************
import sys
import torch

# This one is tricky. We download checkpoint from
# https://api.ngc.nvidia.com/v2/models/nvidia/waveglow_ljs_256channels/versions/2/files/waveglow_256channels_ljs_v2.pt
# and it wants 'glow' to be in PYTHONPATH
sys.path.append('./nemo/collections/tts/parts')

# These four are absolutely required for torch.load
from nemo.collections.tts.parts.waveglow import (
    WaveGlow,
    WN,
    Invertible1x1Conv,
    WaveGlowLoss)


def get_model(model_config, to_cuda):
    model = WaveGlow(**model_config)
    if to_cuda:
        model = model.cuda()
    return model


def load_and_setup_model(checkpoint):
    model_config = dict(
        n_mel_channels=80,
        n_flows=12,
        n_group=8,
        n_early_every=4,
        n_early_size=2,
        WN_config=dict(
            n_layers=8,
            kernel_size=3,
            n_channels=512))
    model = get_model(model_config, to_cuda=True)
    if checkpoint is not None:
        cp_model = torch.load(checkpoint)['model']
        model = WaveGlow.remove_weightnorm(cp_model)
    else:
        model = WaveGlow.remove_weightnorm(model)
    model.eval()
    model.cuda()
    return model

