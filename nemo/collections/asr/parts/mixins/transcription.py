# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import os
import tempfile
import subprocess

from abc import ABC, abstractmethod
from typing import List, Optional

import librosa
import soundfile
import numpy as np
import torch
from torch.utils.data import IterableDataset
from omegaconf import DictConfig, OmegaConf, open_dict
from dataclasses import dataclass, field

import nemo.collections.asr.models as asr_models
from nemo.collections.asr.parts.mixins.asr_adapter_mixins import ASRAdapterModelMixin
from nemo.collections.asr.parts.mixins.streaming import StreamingEncoder
from nemo.collections.asr.parts.utils import asr_module_utils
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.collections.asr.parts.utils.audio_utils import ChannelSelectorType
from nemo.collections.common import tokenizers
from nemo.core.classes import typecheck
from nemo.utils import logging

try:
    import requests
    HAVE_REQUESTS = True
except (ImportError, ModuleNotFoundError):
    HAVE_REQUESTS = False


@dataclass
class TranscriptionConfig:

    channel_selector: ChannelSelectorType = None
    model_stride: int = -1


def ffmpeg_read(bpayload: bytes, sampling_rate: int) -> np.array:
    """
    Helper function to read an audio file through ffmpeg.
    """
    ar = f"{sampling_rate}"
    ac = "1"
    format_for_conversion = "f32le"
    ffmpeg_command = [
        "ffmpeg",
        "-i",
        "pipe:0",
        "-ac",
        ac,
        "-ar",
        ar,
        "-f",
        format_for_conversion,
        "-hide_banner",
        "-loglevel",
        "quiet",
        "pipe:1",
    ]

    try:
        with subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE) as ffmpeg_process:
            output_stream = ffmpeg_process.communicate(bpayload)

        out_bytes = output_stream[0]
        audio = np.frombuffer(out_bytes, np.float32)
        if audio.shape[0] == 0:
            raise ValueError(
                "Soundfile is either not in the correct format or is malformed. Ensure that the soundfile has "
                "a valid audio file extension (e.g. wav, flac or mp3) and is not corrupted. If reading from a remote "
                "URL, ensure that the URL is the full address to **download** the audio file."
            )
    except FileNotFoundError as error:
        # raise ValueError("ffmpeg was not found but is required to load audio files from filename") from error
        logging.warning('ffmpeg was not found but is required to load audio files from filename. Attempting to read via soundfile')
        import soun


    return audio


def rescale_stride(stride, ratio):
    """
    Rescales the stride values from audio space to tokens/logits space.

    (160_000, 16_000, 16_000) -> (2000, 200, 200) for instance.
    """
    # Shape is [B, SEQ] for tokens
    # [B, SEQ, V] for logits

    new_strides = []
    for input_n, left, right in stride:
        token_n = int(round(input_n * ratio))
        left = int(round(left / input_n * token_n))
        right = int(round(right / input_n * token_n))
        new_stride = (token_n, left, right)
        new_strides.append(new_stride)

    return new_strides


def chunk_iter(inputs, preprocessor, chunk_len, stride_left, stride_right, rescale=True, dtype=None):
    inputs_len = inputs.shape[0]
    step = chunk_len - stride_left - stride_right
    for chunk_start_idx in range(0, inputs_len, step):
        chunk_end_idx = chunk_start_idx + chunk_len
        chunk = inputs[chunk_start_idx:chunk_end_idx]
        chunk_len = torch.tensor([chunk.shape[-1]])
        with typecheck.disable_checks():
            processed, processed_len = preprocessor(chunk, chunk_len)
            processed = processed[0]
            processed_len = processed_len[0]
        if dtype is not None:
            processed = processed.to(dtype=dtype)
        _stride_left = 0 if chunk_start_idx == 0 else stride_left
        # all right strides must be full, otherwise it is the last item
        is_last = chunk_end_idx > inputs_len if stride_right > 0 else chunk_end_idx >= inputs_len
        _stride_right = 0 if is_last else stride_right

        chunk_len = chunk.shape[0]
        stride = (chunk_len, _stride_left, _stride_right)
        # if "input_features" in processed:
        #     processed_len = processed["input_features"].shape[-1]
        # elif "input_values" in processed:
        #     processed_len = processed["input_values"].shape[-1]
        print("Chunk len :", chunk_len)
        print("Processed len :", processed_len)
        if processed_len != chunk.shape[-1] and rescale:
            ratio = processed_len / chunk_len
            stride = rescale_stride([stride], ratio)[0]

        output = {'processed': processed, 'processed_len': processed_len}

        if chunk.shape[0] > _stride_left:
            yield {"is_last": is_last, "stride": stride, **output}
        if is_last:
            break


class TranscribeDatasetIterator(IterableDataset):
    def __init__(self, dataset, transcribe_fn, transcribe_fn_kwargs, dataset_batch_size=None):
        """
        Modified from https://github.com/huggingface/transformers/blob/4ab5fb8941a38d172b3883c152c34ae2a0b83a68/src/transformers/pipelines/pt_utils.py

        Roughly equivalent to
        ```
        for item in loader:
            yield infer(item, **params)
        ```

                Arguments:
                    loader (`torch.utils.data.DataLoader` or any iterator):
                        The iterator that will be used to apply `infer` on.
                    infer (any function):
                        The function to apply of each element of `loader`.
                    params (`dict`):
                        The parameters passed to `infer` along with every item
                    loader_batch_size (`int`, *optional*):
                        If specified, the items of `loader` are supposed to come as batch, and are loader_batched here
                        making it roughly behave as


        ```
        for items in loader:
            for i in loader_batch_size:
                item = items[i]
                yield infer(item, **params)
        ```"""
        self.dataset = dataset
        self.transcribe_fn = transcribe_fn
        self.params = transcribe_fn_kwargs
        if dataset_batch_size == 1:
            dataset_batch_size = None
        self.dataset_batch_size = dataset_batch_size

        # Internal bookkeeping
        self._dataset_batch_index = None
        self._dataset_batch_data = None

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        self.iterator = iter(self.dataset)
        return self

    def batch_item(self):
        """
        Return item located at `loader_batch_index` within the current `loader_batch_data`.
        """
        if isinstance(self._dataset_batch_data, torch.Tensor):
            # Batch data is simple tensor, just fetch the slice
            result = self._dataset_batch_data[self._dataset_batch_index]
        else:
            # Batch data is assumed to be BaseModelOutput (or dict)
            loader_batched = {}
            for k, element in self._dataset_batch_data.items():
                if element is None:
                    # This can happen for optional data that get passed around
                    loader_batched[k] = None
                elif isinstance(element[self._dataset_batch_index], torch.Tensor):
                    # Take correct batch data, but make it looked like batch_size=1
                    loader_batched[k] = element[self._dataset_batch_index]
                elif isinstance(element[self._dataset_batch_index], np.ndarray):
                    # Take correct batch data, but make it looked like batch_size=1
                    loader_batched[k] = torch.from_numpy(np.expand_dims(element[self._dataset_batch_index], 0))
                else:
                    # This is typically a list, so no need to `unsqueeze`.
                    loader_batched[k] = element[self._dataset_batch_index]
            # Recreate the element by reusing the original class to make it look
            # batch_size=1
            result = self._dataset_batch_data.__class__(loader_batched)
        self._dataset_batch_index += 1
        return result

    def __next__(self):
        if self._dataset_batch_index is not None and self._dataset_batch_index < self.dataset_batch_size:
            # We are currently unrolling a batch so we just need to return
            # the current item within a batch
            return self.batch_item()

        # We're out of items within a batch
        item = next(self.iterator)
        processed = self.transcribe_fn(item, **self.params)
        # We now have a batch of "inferred things".
        if self.dataset_batch_size is not None:
            # Try to infer the size of the batch
            if isinstance(processed, torch.Tensor):
                first_tensor = processed
            else:
                key = list(processed.keys())[0]
                first_tensor = processed[key]
            if isinstance(first_tensor, list):
                observed_batch_size = len(first_tensor)
            else:
                observed_batch_size = first_tensor.shape[0]
            if 0 < observed_batch_size < self.dataset_batch_size:
                # could be last batch so we can't unroll as many
                # elements.
                self.dataset_batch_size = observed_batch_size
            # Setting internal index to unwrap the batch
            self._dataset_batch_data = processed
            self._dataset_batch_index = 0
            return self.batch_item()
        else:
            # We're not unrolling batches
            return processed


class TranscribeChunkDatasetIterator(TranscribeDatasetIterator):
    def __init__(self, dataset, transcribe_fn, transcribe_fn_kwargs, dataset_batch_size=None):
        """
        Modified from https://github.com/huggingface/transformers/blob/4ab5fb8941a38d172b3883c152c34ae2a0b83a68/src/transformers/pipelines/pt_utils.py

        Roughly equivalent to

        ```
        for iterator in loader:
            for item in iterator:
                yield infer(item, **params)
        ```

                Arguments:
                    loader (`torch.utils.data.DataLoader` or any iterator):
                        The iterator that will be used to apply `infer` on.
                    infer (any function):
                        The function to apply of each element of `loader`.
                    params (`dict`):
                        The parameters passed to `infer` along with every item
        """
        super().__init__(dataset, transcribe_fn, transcribe_fn_kwargs)

    def __iter__(self):
        self.iterator = iter(self.dataset)
        self.subiterator = None
        return self

    def __next__(self):
        if self.subiterator is None:
            "Subiterator None means we haven't started a `preprocess` iterator. so start it"
            self.subiterator = self.transcribe_fn(next(self.iterator), **self.params)
        try:
            # Try to return next item
            processed = next(self.subiterator)
        except StopIteration:
            # When a preprocess iterator ends, we can start lookig at the next item
            # ChunkIterator will keep feeding until ALL elements of iterator
            # all have created their subiterator and have been iterating against.
            #
            # Another way to look at it, is we're basically flattening lists of lists
            # into a single list, but with generators
            self.subiterator = self.transcribe_fn(next(self.iterator), **self.params)
            processed = next(self.subiterator)
        return processed


class TranscribePackIterator(TranscribeDatasetIterator):
    """
    Roughly equivalent to

    ```
    packed =  []
    for item in loader:
        packed.append(item)
        if item["is_last"]:
            yield packed
            packed = []
    ```

        but it also handles cases where `item` are batched (meaning it's a dict of Tensor with first dimension > 1. In
        that case it does

    ```
    packed =  []
    for batch in loader:
        # item is batched
        for item in batch:
            packed.append(item)
            if item["is_last"]:
                yield packed
                packed = []
    ```

        Arguments:
            loader (`torch.utils.data.DataLoader` or any iterator):
                The iterator that will be used to apply `infer` on.
            infer (any function):
                The function to apply of each element of `loader`.
            params (`dict`):
                The parameters passed to `infer` along with every item
            loader_batch_size (`int`, *optional*):
                If specified, the items of `loader` are supposed to come as batch, and are loader_batched here making
                it roughly behave as


    ```
    for items in loader:
        for i in loader_batch_size:
            item = items[i]
            yield infer(item, **params)
    ```"""

    def __iter__(self):
        self.iterator = iter(self.dataset)
        return self

    def __next__(self):
        # Extremely similar to PipelineIterator in its unpacking mechanism
        # BUT, we have an extra required item which is the presence of `is_last`
        # That is because everything is flattened by `PipelineChunkIterator` we
        # need to keep track of how to regroup here in the original `process`
        # boundaries so that `process` and `postprocess` see the same data.

        # This iterator accumulates items (possibly while unbatching) until it
        # its a `is_last` and then just passes it on to the caller.
        is_last = False
        accumulator = []
        if self._dataset_batch_index is not None and self._dataset_batch_index < self.dataset_batch_size:
            while self._dataset_batch_index < self.dataset_batch_size:
                item = self.batch_item()
                is_last = item.pop("is_last")
                accumulator.append(item)
                if is_last:
                    return accumulator

        while not is_last:
            processed = self.transcribe_fn(next(self.iterator), **self.params)
            if self.dataset_batch_size is not None:
                if isinstance(processed, torch.Tensor):
                    first_tensor = processed
                else:
                    key = list(processed.keys())[0]
                    first_tensor = processed[key]
                if isinstance(first_tensor, list):
                    observed_batch_size = len(first_tensor)
                else:
                    observed_batch_size = first_tensor.shape[0]
                if 0 < observed_batch_size < self.dataset_batch_size:
                    # could be last batch so we can't unroll as many
                    # elements.
                    self.dataset_batch_size = observed_batch_size
                self._dataset_batch_data = processed
                self._dataset_batch_index = 0
                while self._dataset_batch_index < self.dataset_batch_size:
                    item = self.batch_item()
                    is_last = item.pop("is_last")
                    accumulator.append(item)
                    if is_last:
                        return accumulator
            else:
                item = processed
                is_last = item.pop("is_last")
                accumulator.append(item)
        return accumulator


class Transcribable(ABC):
    """
    An abstract class for transcribable models.
    """

    @torch.no_grad()
    def transcribe(
        self,
        paths2audio_files: List[str],
        batch_size: int = 4,
        logprobs: bool = False,
        return_hypotheses: bool = False,
        num_workers: int = 0,
        channel_selector: Optional[ChannelSelectorType] = None,
        augmentor: DictConfig = None,
        verbose: bool = True,
    ) -> List[str]:

        if paths2audio_files is None or len(paths2audio_files) == 0:
            return {}

        if return_hypotheses and logprobs:
            raise ValueError(
                "Either `return_hypotheses` or `logprobs` can be True at any given time."
                "Returned hypotheses will contain the logprobs."
            )

        if num_workers is None:
            num_workers = min(batch_size, os.cpu_count() - 1)

        # We will store transcriptions here
        hypotheses = []
        all_hypotheses = []

        # Model's mode and device
        mode = self.training
        device = next(self.parameters()).device
        dither_value = self.preprocessor.featurizer.dither
        pad_to_value = self.preprocessor.featurizer.pad_to

        try:
            self.preprocessor.featurizer.dither = 0.0
            self.preprocessor.featurizer.pad_to = 0
            # Switch model to evaluation mode
            self.eval()
            # Freeze the encoder and decoure_exder modules
            self.encoder.freeze()
            self.decoder.freeze()
            logging_level = logging.get_verbosity()
            logging.set_verbosity(logging.WARNING)
            # Work in tmp directory - will store manifest file there
            with tempfile.TemporaryDirectory() as tmpdir:
                with open(os.path.join(tmpdir, 'manifest.json'), 'w', encoding='utf-8') as fp:
                    for audio_file in paths2audio_files:
                        entry = {'audio_filepath': audio_file, 'duration': 100000, 'text': ''}
                        fp.write(json.dumps(entry) + '\n')

                config = {
                    'paths2audio_files': paths2audio_files,
                    'batch_size': batch_size,
                    'temp_dir': tmpdir,
                    'num_workers': num_workers,
                    'channel_selector': channel_selector,
                }

                if augmentor:
                    config['augmentor'] = augmentor

                temporary_datalayer = self._setup_transcribe_dataloader(config)
                for test_batch in tqdm(temporary_datalayer, desc="Transcribing", disable=not verbose):
                    logits, logits_len, greedy_predictions = self.forward(
                        input_signal=test_batch[0].to(device), input_signal_length=test_batch[1].to(device)
                    )

                    if logprobs:
                        # dump log probs per file
                        for idx in range(logits.shape[0]):
                            lg = logits[idx][: logits_len[idx]]
                            hypotheses.append(lg.cpu().numpy())
                    else:
                        current_hypotheses, all_hyp = self.decoding.ctc_decoder_predictions_tensor(
                            logits, decoder_lengths=logits_len, return_hypotheses=return_hypotheses,
                        )
                        logits = logits.cpu()

                        if return_hypotheses:
                            # dump log probs per file
                            for idx in range(logits.shape[0]):
                                current_hypotheses[idx].y_sequence = logits[idx][: logits_len[idx]]
                                if current_hypotheses[idx].alignments is None:
                                    current_hypotheses[idx].alignments = current_hypotheses[idx].y_sequence

                        if all_hyp is None:
                            hypotheses += current_hypotheses
                        else:
                            hypotheses += all_hyp

                    del greedy_predictions
                    del logits
                    del test_batch
        finally:
            # set mode back to its original value
            self.train(mode=mode)
            self.preprocessor.featurizer.dither = dither_value
            self.preprocessor.featurizer.pad_to = pad_to_value
            if mode is True:
                self.encoder.unfreeze()
                self.decoder.unfreeze()
            logging.set_verbosity(logging_level)

        return hypotheses

    def _transcribe_input_processing(self):
        pass

    def _transcribe_on_begin(self):
        pass

    def _transcribe_prepare_dataloader(self):
        pass

    def _transcribe_loop(self):
        pass

    def _transcribe_forward(self):
        pass

    def _transcribe_output_processing(self):
        pass

    def _transcribe_on_end(self):
        pass

    def _transcribe_preprocess_single_file(self, inputs, chunk_length_s=0, stride_length_s=None,
                                           transcribe_cfg: TranscriptionConfig = None):
        assert transcribe_cfg is not None, "Transcription config must be passed to preprocess"

        if isinstance(inputs, str):
            if inputs.startswith("http://") or inputs.startswith("https://"):
                # We need to actually check for a real protocol, otherwise it's impossible to use a local file
                # like http_huggingface_co.png
                if HAVE_REQUESTS:
                    inputs = requests.get(inputs).content
                else:
                    raise ImportError(
                        "You need to install the requests module to use HTTP(S) URLs for inference. "
                        "You can install it through `pip install requests`."
                    )
            else:
                with open(inputs, "rb") as f:
                    inputs = f.read()

        if isinstance(inputs, bytes):
            # Access sample rate from config
            inputs = ffmpeg_read(inputs, self.cfg.sample_rate)

        stride = None
        extra = {}
        if isinstance(inputs, dict):
            stride = inputs.pop("stride", None)
            # Accepting `"array"` which is the key defined in `datasets` for
            # better integration
            if not ("sample_rate" in inputs and ("raw" in inputs or "array" in inputs)):
                raise ValueError(
                    "When passing a dictionary to transcribe(), the dict needs to contain a "
                    '"raw" key containing the numpy array representing the audio and a "sample_rate" key, '
                    "containing the sample_rate associated with that array"
                )

            _inputs = inputs.pop("raw", None)
            if _inputs is None:
                # Remove path which will not be used from `datasets`.
                inputs.pop("path", None)
                _inputs = inputs.pop("array", None)
            in_sampling_rate = inputs.pop("sample_rate")
            extra = inputs
            inputs = _inputs
            if in_sampling_rate != self.cfg.sample_rate:
                inputs = librosa.resample(y=inputs, orig_sr=in_sampling_rate, target_sr=self.cfg.sample_rate)
                ratio = self.cfg.sample_rate / in_sampling_rate
            else:
                ratio = 1
            # if stride is not None:
            #     if stride[0] + stride[1] > inputs.shape[0]:
            #         raise ValueError("Stride is too large for input")
            #
            #     # Stride needs to get the chunk length here, it's going to get
            #     # swallowed by the `feature_extractor` later, and then batching
            #     # can add extra data in the inputs, so we need to keep track
            #     # of the original length in the stride so we can cut properly.
            #     stride = (inputs.shape[0], int(round(stride[0] * ratio)), int(round(stride[1] * ratio)))
        if not isinstance(inputs, np.ndarray):
            raise ValueError(f"We expect a numpy ndarray as input, got `{type(inputs)}`")

        if inputs.ndim != 1:
            if transcribe_cfg.channel_selector is None:
                raise ValueError(
                    f"Input is multi-channel but no `channel_selector` was passed. "
                    f"Please pass a `channel_selector` to select a single channel from the input."
                )
            inputs = inputs[transcribe_cfg.channel_selector]

        if not torch.is_tensor(inputs):
            inputs = torch.from_numpy(inputs)

        if chunk_length_s:
            if stride_length_s is None:
                stride_length_s = chunk_length_s / 6

            if isinstance(stride_length_s, (int, float)):
                stride_length_s = [stride_length_s, stride_length_s]

            # XXX: Carefuly, this variable will not exist in `seq2seq` setting.
            # Currently chunking is not possible at this level for `seq2seq` so
            # it's ok.

            if 'model_stride' in self.cfg and self.cfg.model_stride > 0:
                align_to = self.cfg.model_stride
            else:
                if transcribe_cfg.model_stride < 0:
                    raise ValueError("When using chunk_length_s, model_stride must be set in the transcription config")
                align_to = transcribe_cfg.model_stride
                self.cfg.model_stride = transcribe_cfg.model_stride

            chunk_len = int(round(chunk_length_s * self.cfg.sample_rate / align_to) * align_to)
            stride_left = int(round(stride_length_s[0] * self.cfg.sample_rate / align_to) * align_to)
            stride_right = int(round(stride_length_s[1] * self.cfg.sample_rate / align_to) * align_to)

            if chunk_len < stride_left + stride_right:
                raise ValueError("Chunk length must be superior to stride length")

            # make sure that
            for item in chunk_iter(
                    inputs, self.preprocessor, chunk_len, stride_left, stride_right, rescale=True, dtype=None
            ):
                yield item
        else:
            with typecheck.disable_checks():
                inputs_len = torch.tensor([inputs.shape[0]])
                processed, processed_len = self.preprocessor(inputs, inputs_len)

                output = {'processed': processed, 'processed_len': processed_len}

            # if self.torch_dtype is not None:
            #     processed = processed.to(dtype=self.torch_dtype)
            if stride is not None:
                # if self.type == "seq2seq":
                #     raise ValueError("Stride is only usable with CTC models, try removing it !")

                output["stride"] = stride
            yield {"is_last": True, **output, **extra}