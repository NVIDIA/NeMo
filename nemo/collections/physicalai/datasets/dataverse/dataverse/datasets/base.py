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

from abc import ABC, abstractmethod
from typing import Any, List, Optional

from . import DataField


class BaseDataset(ABC):
    """
    Base class for all datasets.
    Note that this is not directly wrap-able by a dataloader. It is meant to be
    subclassed / included in another dataset class.
    """

    def __init__(self):
        pass

    @abstractmethod
    def available_data_fields(self) -> List[DataField]:
        """
        Return a list of available data fields in the dataset.
        """
        pass

    @abstractmethod
    def num_videos(self) -> int:
        """
        Returns:
            Number of videos in the dataset.
        """
        pass

    @abstractmethod
    def num_views(self, video_idx: int) -> int:
        """
        Args:
            video_idx: Index of the video.

        Returns:
            Number of views in the video.
        """
        pass

    @abstractmethod
    def num_frames(self, video_idx: int, view_idx: int = 0) -> int:
        """
        Args:
            video_idx: Index of the video.
            view_idx: Index of the view.

        Returns:
            Number of frames in the given view.
        """
        pass

    def read(
        self,
        video_idx: int,
        frame_idxs: Optional[List[int]] = None,
        view_idxs: Optional[List[int]] = None,
        data_fields: Optional[List[DataField]] = None,
    ) -> dict[DataField, Any]:
        """
        Read data from the dataset.
        Args:
            video_idx: Index of the video.
            frame_idxs: List of frame indices.
            view_idxs: List of view indices.
            data_fields: List of data fields to read. If None, read all data fields.

        Example:
            if frame_idxs is None, view_idxs is None, read all frames from all views.
            if frame_idxs is not None, view_idxs is None, read frames from the first view.
            if frame_idxs is None, view_idxs is not None, read all frames from the specified views.
            if frame_idxs is not None, view_idxs is not None, read frames from the specified views.

        Returns:
            A dictionary mapping data fields to their values.
        """

        if data_fields is None:
            data_fields = self.available_data_fields()

        if frame_idxs is None:
            # Frame not provided, default read all frames.
            if view_idxs is None:
                view_iterator = range(self.num_views(video_idx))
            else:
                view_iterator = view_idxs

            new_frame_idxs, new_view_idxs = [], []
            for view_idx in view_iterator:
                num_frames = self.num_frames(video_idx, view_idx)
                new_frame_idxs.extend(list(range(num_frames)))
                new_view_idxs.extend([view_idx] * num_frames)
            frame_idxs, view_idxs = new_frame_idxs, new_view_idxs

        elif view_idxs is None:
            # View not provided, but frame is provided, we only read the first view.
            view_idxs = [0] * len(frame_idxs)

        else:
            # Both frame_idxs and view_idxs provided, do sanity check.
            assert len(frame_idxs) == len(view_idxs), "Frame and view indices must match."

        return self._read_data(
            video_idx=video_idx,
            frame_idxs=frame_idxs,
            view_idxs=view_idxs,
            data_fields=data_fields,
        )

    @abstractmethod
    def _read_data(
        self,
        video_idx: int,
        frame_idxs: List[int],
        view_idxs: List[int],
        data_fields: List[DataField],
    ) -> dict[DataField, Any]:
        pass
