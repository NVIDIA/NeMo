# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import argparse
import contextlib
import itertools
import json
from collections import UserList
from typing import Optional

import torch
import tqdm
import time
import numpy as np
from torch import nn
from scipy.stats import norm

from nemo.collections.tts.models import MixerTTSModel


def parse_args() -> argparse.Namespace:
    """Parses args from CLI."""
    parser = argparse.ArgumentParser(description='Mixer-TTS Benchmark')
    parser.add_argument('--manifest-path', type=str, required=True)
    parser.add_argument('--model-ckpt-path', type=str, required=True)
    parser.add_argument('--without-matching', action='store_true', default=False)
    parser.add_argument('--torchscript', action='store_true', default=False)
    parser.add_argument('--amp-half', action='store_true', default=False)
    parser.add_argument('--amp-autocast', action='store_true', default=False)
    parser.add_argument('--n-chars', type=int, default=128)
    parser.add_argument('--n-samples', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--warmup-repeats', type=int, default=3)
    parser.add_argument('--n-repeats', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--cudnn-benchmark', action='store_true', default=False)
    return parser.parse_args()


def make_data(manifest_path: str, n_chars: Optional[int] = None, n_samples: Optional[int] = None):
    """Makes data source and returns batching functor and total number of samples."""

    if n_chars is None:
        raise ValueError("Unfixed number of input chars is unsupported for now.")

    raw_text_data = []
    with open(manifest_path, 'r') as f:
        for line in f.readlines():
            line_data = json.loads(line)
            raw_text_data.append(line_data['text'])

    if n_samples is None:
        n_samples = len(raw_text_data)

    raw_text_stream = itertools.cycle(raw_text_data)
    data, raw_text_buffer = [], []
    while len(data) < n_samples:
        raw_text_buffer.append(next(raw_text_stream))
        raw_text_from_buffer = ' '.join(raw_text_buffer)
        if len(raw_text_from_buffer) >= n_chars:
            data.append(dict(raw_text=raw_text_from_buffer[:n_chars]))
            raw_text_buffer.clear()

    # This is probably redundant as all samples are of the same length.
    data.sort(key=lambda d: len(d['raw_text']), reverse=True)  # Bigger samples are more important.

    data = {k: [s[k] for s in data] for k in data[0]}
    raw_text_data = data['raw_text']
    total_samples = len(raw_text_data)

    def batching(batch_size):
        """<batch size> => <batch generator>"""
        for i in range(0, len(raw_text_data), batch_size):
            yield dict(raw_text=raw_text_data[i : i + batch_size])

    return batching, total_samples


def load_and_setup_model(
    ckpt_path: str,
    amp: bool = False,
    torchscript: bool = False,
) -> nn.Module:
    """Loads and setup Mixer-TTS model."""

    model = MixerTTSModel.load_from_checkpoint(ckpt_path)

    if amp:
        model = model.half()

    if torchscript:
        model = torch.jit.script(model)

    model.eval()

    return model


class MeasureTime(UserList):
    """Convenient class for time measurement."""

    def __init__(self, *args, cuda=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.cuda = cuda

    def __enter__(self):
        if self.cuda:
            torch.cuda.synchronize()
        self.t0 = time.perf_counter()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self.cuda:
            torch.cuda.synchronize()
        self.append(time.perf_counter() - self.t0)


def main():
    """Launches TTS benchmark."""

    args = parse_args()

    batching, total_samples = make_data(args.manifest_path, args.n_chars, args.n_samples)

    model = load_and_setup_model(args.model_ckpt_path, args.amp_half, args.torchscript)
    model.to(args.device)

    torch.backends.cudnn.benchmark = args.cudnn_benchmark  # noqa

    def switch_amp_on():
        """Switches AMP on."""
        return (
            torch.cuda.amp.autocast(enabled=True)
            if args.amp_autocast
            else contextlib.nullcontext()
        )

    def batches(batch_size):
        """Batches generator."""
        for b in tqdm.tqdm(
            iterable=batching(batch_size),
            total=(total_samples // batch_size) + int(total_samples % batch_size),
            desc='batches',
        ):
            yield b

    # Warmup
    for _ in tqdm.trange(args.warmup_repeats, desc='warmup'):
        with torch.no_grad(), switch_amp_on():
            for batch in batches(args.batch_size):
                _ = model.generate_spectrogram(
                    raw_texts=batch['raw_text'],
                    without_matching=args.without_matching,
                )

    sample_rate = model.cfg.train_ds.dataset.sample_rate
    hop_length = model.cfg.train_ds.dataset.hop_length
    gen_measures = MeasureTime(cuda=(args.device != 'cpu'))
    all_letters, all_frames = 0, 0
    all_utterances, all_samples = 0, 0
    for _ in tqdm.trange(args.n_repeats, desc='repeats'):
        for batch in batches(args.batch_size):
            with torch.no_grad(), switch_amp_on(), gen_measures:
                mel = model.generate_spectrogram(
                    raw_texts=batch['raw_text'],
                    without_matching=args.without_matching,
                )

            all_letters += sum(len(t) for t in batch['raw_text'])  # <raw text length>
            # TODO(stasbel): Actually, this need to be more precise as samples are of different length?
            all_frames += mel.size(0) * mel.size(1)  # <batch size> * <mel length>

            all_utterances += len(batch['raw_text'])  # <batch size>
            # TODO(stasbel): Same problem as above?
            # <batch size> * <mel length> * <hop length> = <batch size> * <audio length>
            all_samples += mel.size(0) * mel.size(1) * hop_length

    gm = np.sort(np.asarray(gen_measures))
    results = {
        'avg_letters/s': all_letters / gm.sum(),
        'avg_frames/s': all_frames / gm.sum(),
        'avg_latency': gm.mean(),
        'all_samples': all_samples,
        'all_utterances': all_utterances,
        'without_matching': args.without_matching,
        'avg_RTF': all_samples / (all_utterances * gm.mean() * sample_rate),
        '90%_latency': gm.mean() + norm.ppf((1.0 + 0.90) / 2) * gm.std(),
        '95%_latency': gm.mean() + norm.ppf((1.0 + 0.95) / 2) * gm.std(),
        '99%_latency': gm.mean() + norm.ppf((1.0 + 0.99) / 2) * gm.std(),
    }
    for k, v in results.items():
        print(f'{k}: {v}')


if __name__ == '__main__':
    main()