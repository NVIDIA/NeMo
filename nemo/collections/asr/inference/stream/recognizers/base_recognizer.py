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


from abc import abstractmethod
from typing import Any, Dict, Iterable, List, Optional

from nemo.collections.asr.inference.stream.framing.multi_stream import ContinuousBatchedRequestStreamer
from nemo.collections.asr.inference.stream.framing.request import FeatureBuffer, Frame, Request
from nemo.collections.asr.inference.stream.framing.request_options import ASRRequestOptions
from nemo.collections.asr.inference.stream.recognizers.recognizer_interface import RecognizerInterface
from nemo.collections.asr.inference.utils.progressbar import ProgressBar
from nemo.collections.asr.inference.utils.word import Word


class RecognizerOutput:
    """
    Class to store the output of the recognizer.
    """

    def __init__(self, texts: List[str] = None, words: List[List[Word]] = None):
        if texts is None and words is None:
            raise ValueError("At least one of the 'texts' or 'words' should be provided.")
        self.texts = texts
        self.words = words


class BaseRecognizer(RecognizerInterface):
    """
    Base class for all recognizers.
    """

    def __init__(self):
        """Initialize state pool to store the state for each stream"""
        self._state_pool: Dict[int, Any] = {}

    def get_state(self, stream_id: int) -> Any:
        """Retrieve state for a given stream ID."""
        return self._state_pool.get(stream_id, None)

    def get_states(self, stream_ids: Iterable[int]) -> List[Any]:
        """Retrieve states for a list of stream IDs."""
        return [self.get_state(stream_id) for stream_id in stream_ids]

    def delete_state(self, stream_id: int) -> None:
        """Delete the state from the state pool."""
        if stream_id in self._state_pool:
            del self._state_pool[stream_id]

    def delete_states(self, stream_ids: Iterable[int]) -> None:
        """Delete states for a list of stream IDs."""
        for stream_id in stream_ids:
            self.delete_state(stream_id)

    def init_state(self, stream_id: int, options: ASRRequestOptions) -> Any:
        """Initialize the state of the stream"""
        if stream_id not in self._state_pool:
            state = self.create_state(options)
            self._state_pool[stream_id] = state
        return self._state_pool[stream_id]

    def reset_session(self) -> None:
        """Reset the frame buffer and internal state pool"""
        self._state_pool.clear()

    def open_session(self) -> None:
        """Start a new session by resetting the internal state pool"""
        self.reset_session()

    def close_session(self) -> None:
        """Close the session by resetting the internal state pool"""
        self.reset_session()

    @abstractmethod
    def transcribe_step_for_frames(self, frames: List[Frame]) -> None:
        """Transcribe a step for frames"""
        pass

    @abstractmethod
    def transcribe_step_for_feature_buffers(self, fbuffers: List[FeatureBuffer]) -> None:
        """Transcribe a step for feature buffers"""
        pass

    @abstractmethod
    def get_request_generator(self) -> ContinuousBatchedRequestStreamer:
        """Return the request generator."""
        pass

    @abstractmethod
    def get_sep(self) -> str:
        """Return the separator for the text postprocessor."""
        pass

    def transcribe_step(self, requests: List[Request]) -> None:
        """Transcribe a step"""
        if isinstance(requests[0], Frame):
            self.transcribe_step_for_frames(frames=requests)
        elif isinstance(requests[0], FeatureBuffer):
            self.transcribe_step_for_feature_buffers(fbuffers=requests)
        else:
            raise ValueError(f"Invalid request type: {type(requests[0])}")

    def run(
        self,
        audio_filepaths: List[str],
        options: List[ASRRequestOptions] = None,
        progress_bar: Optional[ProgressBar] = None,
    ) -> RecognizerOutput:
        """
        Orchestrates reading from audio_filepaths in a streaming manner,
        transcribes them, and packs the results into a RecognizerOutput.
        Args:
            audio_filepaths: List of audio filepaths to transcribe.
            options: List of RequestOptions for each stream.
            progress_bar: Progress bar to show the progress. Default is None.
        Returns:
            RecognizerOutput: A dataclass containing transcriptions and words.
        """
        if progress_bar is not None and not isinstance(progress_bar, ProgressBar):
            raise ValueError("progress_bar must be an instance of ProgressBar.")

        if options is None:
            # Use default options if not provided
            options = [ASRRequestOptions() for _ in audio_filepaths]

        if len(options) != len(audio_filepaths):
            raise ValueError("options must be the same length as audio_filepaths")

        request_generator = self.get_request_generator()
        request_generator.set_audio_filepaths(audio_filepaths, options)
        request_generator.set_progress_bar(progress_bar)

        self.open_session()
        for requests in request_generator:
            for request in requests:
                if request.is_first:
                    self.init_state(request.stream_id, request.options)
            self.transcribe_step(requests)
        output = self.pack_output()
        self.close_session()
        return output

    def pack_output(self) -> RecognizerOutput:
        """Pack the output from the internal state pool."""
        texts, words = [], []
        for stream_id in sorted(self._state_pool):
            state = self.get_state(stream_id)
            # by default, we will store final words in pnc_words
            # and itn-ed words in itn_words
            attr_name = "itn_words" if state.options.enable_itn else "pnc_words"
            state_words = getattr(state, attr_name)
            state_text = self.get_sep().join(word.text for word in state_words)
            texts.append(state_text)
            words.append(state_words)
        return RecognizerOutput(texts=texts, words=words)
