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


from typing import Dict, List

from nemo.collections.asr.inference.stream.state.state import StreamingState


class CacheAwareStreamingState(StreamingState):
    """
    State of the cache aware CTC/RNNT streaming recognizers
    """

    def reset(self) -> None:
        """
        Reset the state
        """
        super().reset()
        # label_buffer will be used to detect EoU
        self.label_buffer = []
        self.label_buffer_size = 0
        self.offset = 0

    def set_offset(self, offset: int) -> None:
        """
        Set the offset
        Args:
            offset: (int) offset
        """
        self.offset = offset

    def setup_label_buffer(self, label_buffer_size: int, blank_id: int) -> None:
        """
        Set up the label buffer
        Args:
            label_buffer_size: (int) size of the label buffer
            blank_id: (int) blank id
        """
        self.label_buffer_size = label_buffer_size
        self.label_buffer = [blank_id] * self.label_buffer_size

    def update_label_buffer(self, labels: list[int]) -> None:
        """
        Update the label buffer
        Args:
            labels: (List[int]) list of labels
        """
        shift = len(labels)
        self.label_buffer[:-shift] = self.label_buffer[shift:].copy()
        self.label_buffer[-shift:] = labels.copy()

    def get_label_buffer(self) -> List[int]:
        """
        Get the current label buffer
        Returns:
            list[int]: current state of the label buffer
        """
        return self.label_buffer.copy()

    def update_state(self, completed_output: Dict, eou_detected: bool) -> None:
        """
        Update the state with the completed output
        Args:
            completed_output: (Dict) completed output
            eou_detected: (bool) is EoU detected
        """

        if len(completed_output) == 0 or len(completed_output["tokens"]) == 0:
            return

        timesteps = completed_output["timesteps"]
        for i, t in enumerate(timesteps):
            timesteps[i] = t + self.global_offset

        # we will not perform overlap aware merging of the tokens for CacheAware Models
        # It is too error-prone to do this in the streaming mode -> skip=0
        self._update_state(completed_output, skip=0)
        self.eou_detected_before = eou_detected
