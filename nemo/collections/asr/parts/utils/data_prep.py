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

import lhotse.dataset
import torch
from lhotse import CutSet


class ToAudio(torch.utils.data.Dataset):
    def __init__(self, return_dict=True):
        self.return_dict = return_dict

    def __getitem__(self, cuts: CutSet):
        audios, audio_lens = cuts.load_audio(collate=True)
        text = [cut.supervisions[0].text for cut in cuts]

        tokens = torch.zeros(len(text), 0, dtype=torch.int64)
        token_lens = torch.zeros(len(text), dtype=torch.int64)

        if self.return_dict:
            return {"cuts": cuts, "audios": audios, "audio_lens": audio_lens, "text": text}
        else:
            return audios, audio_lens, tokens, token_lens


def get_lhotse_dataloader(cuts, batch_size, return_dict=False,num_workers=1):
    dloader = torch.utils.data.DataLoader(
        dataset=ToAudio(return_dict=return_dict),
        sampler=lhotse.dataset.DynamicCutSampler(cuts, max_cuts=batch_size),
        num_workers=num_workers,
        batch_size=None,
    )
    return dloader
