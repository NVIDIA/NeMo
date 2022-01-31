# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import json
import os
from collections import namedtuple
from pathlib import Path
from typing import Any, List, Tuple

from nemo.collections.asr.data.audio_to_text_dataset import ASRPredictionWriter
from nemo.utils import logging

FrameCtmUnit = namedtuple('FrameCtmUnit', ['label', 'start_frame', 'length', 'probability'])


class ASRCTMPredictionWriter(ASRPredictionWriter):
    def __init__(self, dataset, output_file: str, output_ctm_dir: str, time_per_frame: float):
        super().__init__(dataset, output_file)
        self.output_ctm_dir = output_ctm_dir
        self.time_per_frame = time_per_frame
        os.makedirs(self.output_ctm_dir, exist_ok=True)

    def write_ctm(self, name, filepath, frameCtmUnits):
        with open(filepath, "tw", encoding="utf-8") as f:
            for unit in frameCtmUnits:
                f.write(
                    f"{name} 1 {unit.start_frame * self.time_per_frame :.3f} {unit.length * self.time_per_frame :.3f} {unit.label} {unit.probability :1.2f}\n"
                )

    def write_on_batch_end(
        self,
        trainer,
        pl_module: 'LightningModule',
        prediction: Tuple[int, List[FrameCtmUnit]],
        batch_indices: List[int],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ):
        for sample_id, units in prediction:
            sample = self.dataset.get_manifest_sample(sample_id)
            with_ctm = True
            if len(units) == 0:
                logging.warning(
                    f"""Do not producing CTM output for item `{sample.audio_file}`.
                                Check if text is empty or if duration is too short: `{sample.text_raw}`, {sample.duration}"""
                )
                with_ctm = False
            item = {}
            item["audio_filepath"] = sample.audio_file
            item["duration"] = sample.duration
            item["text"] = sample.text_raw
            if with_ctm:
                utt_name = Path(sample.audio_file).stem
                ctm_filepath = os.path.join(self.output_ctm_dir, utt_name) + ".ctm"
                self.write_ctm(utt_name, ctm_filepath, units)
                item["ctm_filepath"] = ctm_filepath
            else:
                item["ctm_filepath"] = ""
            self.outf.write(json.dumps(item) + "\n")
            self.samples_num += 1
        return
