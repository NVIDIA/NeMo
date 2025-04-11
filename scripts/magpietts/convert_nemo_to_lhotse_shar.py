# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

"""
Example entry in `/home/xueyang/workspace/pretrain/data_prep/hifitts2/manifests/train_manifest_withContextAudioMinDur3.json`
{
  "audio_filepath": "100/2315/100_2315_sea_fairies_0812_librivox-01_baum_sea_fairies_0.flac",
  "duration": 6.2,
  "speaker": "| Language:en Dataset:HiFiTTS2 Speaker:100 |",
  "text": "THE oceans are big and broad. I believe two thirds of the earth's surface is covered with water.",
  "normalized_text": "THE oceans are big and broad. I believe two thirds of the earth's surface is covered with water.",
  "text_source": "book",
  "bandwidth": 13092,
  "snr1": 41.27,
  "snr2": 41.05,
  "snr3": 32.58,
  "snr4": 22.28,
  "is_segmented": true,
  "wer": 0,
  "cer": 0,
  "ins": 0,
  "del": 0,
  "sub": 0,
  "speaker_count": 1,
  "chapter_id": "01",
  "context_speaker_similarity": 0.9059218168258667,
  "context_audio_filepath": "100/2315/100_2315_sea_fairies_0812_librivox-01_baum_sea_fairies_1.flac",
  "context_audio_duration": 6.08
}

Goal: avoid to save inodes quota by tarring individual files for audio codecs, audio waveforms and/or speaker embeddings.
      We can decide if remove audio waveform files later.

Examples of shard files:
$ tree data-shar-train/
 - cuts.000000.jsonl.gz: add all and exclude unnecessary fields.
 - codes_21fpsCausalDecoder.000000.tar
 - recording.000000.tar: not used during training, but worth to tar them so save inodes quota and for future applications.
 - context_codes_21fpsCausalDecoder.000000.tar
 - context_recording.000000.tar
 - context_spk_embed.000000.tar (optional): speaker embedding is not used during training/validation.
 - spk_embed.000000.tar (optional): speaker embedding is not used during training/validation.
"""

import argparse
import os
import re
from functools import partial
from pathlib import Path

import lightning.pytorch as pl
import torch
from lhotse import MonoCut, Recording, SupervisionSegment
from lhotse.shar.writers.shar import AudioTarWriter, SharWriter
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import BasePredictionWriter
from lightning.pytorch.strategies import DDPStrategy
from torch.utils.data import DataLoader, Dataset

from nemo.collections.asr.parts.preprocessing.segment import AudioSegment
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest
from nemo.collections.tts.models import AudioCodecModel

MAYBE_EXTRA_METADATA_IN_MANIFEST = ["normalized_text", "speaker_count", "cer", "wer"]


def check_speaker_format(item: str):
    # enforce the format as example like "| Language:en Dataset:HiFiTTS Speaker:9136_other |".
    pattern = r"\| Language:\w+ Dataset:[\w\d\W]+ Speaker:[\w\d\W]+ \|"
    return bool(re.match(pattern, item))


def get_recording_id(audio_base_dir: str, path: Path):
    # the recording id is defined as the concatenation of relative audio filepath with a hyphen delimiter.
    return path.relative_to(audio_base_dir).with_suffix("").as_posix().replace("/", "-")


class SharPredictionWriter(BasePredictionWriter):
    def __init__(
        self,
        output_dir: str,
        codec_model_name: str,
        codec_frame_rate: float,
        audio_base_dir: str,
        fields: dict,
        shard_size: int = 1000,
    ):
        super().__init__(write_interval="batch")
        self.output_dir = output_dir
        self.codec_model_name = codec_model_name
        self.codec_frame_rate = codec_frame_rate
        self.fields = fields
        self.shard_size = shard_size
        self.batch_counter = 0
        self.shar_writer = None
        self.context_recording_writer = None
        self.is_initialized = False
        self.recording_id_fn = partial(get_recording_id, audio_base_dir)

        # Add a buffer with the shard size to accumulate cuts before writing to disk.
        self.cuts_buffer = list()
        self.buffer_size = shard_size

    def setup(self, trainer, pl_module, stage=None):
        if not self.is_initialized:
            # Only initialize the SharWriter and AudioTarWriter on rank 0
            if trainer.global_rank == 0:
                os.makedirs(self.output_dir, exist_ok=True)

                # Initialize SharWriter
                self.shar_writer = SharWriter(
                    output_dir=self.output_dir, fields=self.fields, shard_size=self.shard_size
                )
                self.shar_writer.__enter__()

                # Initialize AudioTarWriter to store context recording as a workaround.
                # TODO @xueyang: Without this, the process would be blocked because,
                #  When target duration is specified in MonoCut, the error will happen iff context duration < target duration
                #  mostly because the cut tries to trim the context_recording to the same duration as target. No errors
                #  were observed when context duration > target duration. Ref is https://nvidia.slack.com/archives/D068LR4TWUW/p1741817511544239
                self.context_recording_writer = AudioTarWriter(
                    pattern=os.path.join(self.output_dir, "context_recording.%06d.tar"),
                    shard_size=self.shard_size,
                    format="flac",
                )
                self.context_recording_writer.__enter__()

            self.is_initialized = True

    def write_on_batch_end(self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx):
        # prepare cuts from each rank
        pred_cuts = self.convert_prediction_to_cuts(prediction)
        # Gather predictions from all ranks
        gathered_objects = [None] * trainer.world_size
        torch.distributed.all_gather_object(gathered_objects, pred_cuts)

        # Only rank 0 writes to disk
        if trainer.global_rank == 0:
            for _pred_cuts, _context_recordings in gathered_objects:
                if _pred_cuts is None or _context_recordings is None:
                    raise RuntimeError("Received None from all_gather_object")

                # Buffer the cuts
                self.cuts_buffer.extend(list(zip(_pred_cuts, _context_recordings)))

                # Write when buffer is full
                if len(self.cuts_buffer) >= self.buffer_size:
                    self._write_buffer()

    def _write_buffer(self):
        """Write accumulated cuts from buffer"""
        for cut, recording in self.cuts_buffer:
            self.shar_writer.write(cut)
            self.context_recording_writer.write(
                key=cut.id,
                value=recording.load_audio(),
                sampling_rate=recording.sampling_rate,
                manifest=recording,
            )
            self.batch_counter += 1

        # Clear the buffer
        self.cuts_buffer = list()

    def convert_prediction_to_cuts(self, prediction):
        # Extra useful metadata may exist in some manifests, so better to keep them for future usage.
        meta_fields = {
            meta_field: prediction[meta_field]
            for meta_field in MAYBE_EXTRA_METADATA_IN_MANIFEST
            if meta_field in prediction
        }

        # This should convert predictions to Cut objects for Lhotse
        cuts = list()
        context_recordings = list()

        # batch process recordings and codes here.
        # target recording
        target_recordings = [
            Recording.from_file(
                path=audio_filepath,
                recording_id=self.recording_id_fn,
            )
            for audio_filepath in prediction["target_audio_filepath"]
        ]
        context_recordings = [
            Recording.from_file(
                path=audio_filepath,
                recording_id=self.recording_id_fn,
            )
            for audio_filepath in prediction["context_audio_filepath"]
        ]

        # Create supervisions in batch
        supervisions = [
            SupervisionSegment(
                id=f"sup-{rec.id}",
                recording_id=rec.id,
                start=0.0,
                duration=rec.duration,
                channel=0,
                text=text,
                speaker=spk,
                language=lang,
                custom={key: val[idx] for key, val in meta_fields.items()} if meta_fields else None,
            )
            for idx, (rec, text, spk, lang) in enumerate(
                zip(target_recordings, prediction["text"], prediction["speaker"], prediction["language"])
            )
        ]

        # Create cuts in batch
        # TODO @xueyang: should file a bug report to `attach_tensor` function. When `temporal_dim=-1`, the tensor is not
        # attached correctly. For example, I found that `cuts[0].codes_21fpsCausalDecoder.load()` and
        # `cuts[0].load_custom("codes_21fpsCausalDecoder")` returns different arrays, with different shapes. But the former
        # returned expected (8,5) shape, while the latter returned (5,5). I also find that, after write shar files, and
        # when i load codes using `CutSet.from_shar()` and no matter which load functions I used, they are all shape of (5,5)
        # instead of (8,5). In any case, using default `temporal_dim` and `frame_shift` addressed this issue.
        cuts = [
            MonoCut(
                id=f"cut-{rec.id}",
                start=0.0,
                duration=rec.duration,
                recording=rec,
                channel=0,
                supervisions=[sup],
                custom={"context_recording": context_rec},
            ).attach_tensor(
                name=f"codes_{self.codec_model_name}",
                data=target_code,
                # temporal_dim=1,
                # frame_shift=1 / self.codec_frame_rate
            ).attach_tensor(
                name=f"context_codes_{self.codec_model_name}",
                data=context_code,
                # temporal_dim=1,
                # frame_shift=1 / self.codec_frame_rate
            )
            for rec, sup, context_rec, target_code, context_code in zip(
                target_recordings,
                supervisions,
                context_recordings,
                prediction["target_codes"],
                prediction["context_codes"],
            )
        ]

        return cuts, context_recordings

    def teardown(self, trainer, pl_module, stage=None):
        # Wait for rank 0 to finish writing
        if trainer.world_size > 1:
            torch.distributed.barrier()

        # Close the SharWriter and AudioTarWriter on rank 0
        if trainer.global_rank == 0:
            # Write any remaining cuts in the buffer before closing
            if self.cuts_buffer:
                self._write_buffer()

            if self.context_recording_writer is not None:
                self.context_recording_writer.close()
            if self.shar_writer is not None:
                self.shar_writer.close()


class AudioDataset(Dataset):
    def __init__(self, manifest: str, audio_base_dir: str, sample_rate: int = 22050, pad_multiple: int = 1024):
        self.audio_base_dir = audio_base_dir
        self.sample_rate = sample_rate
        self.pad_multiple = pad_multiple
        self.items = read_manifest(manifest)

    def __len__(self):
        return len(self.items)

    def get_wav_from_filepath(self, file_path: str):
        features = AudioSegment.segment_from_file(
            file_path,
            target_sr=self.sample_rate,
            n_segments=-1,
            trim=False,
        )
        audio = torch.tensor(features.samples)
        audio = torch.nn.functional.pad(audio, (0, self.pad_multiple - audio.size(0) % self.pad_multiple), value=0)
        audio_length = torch.tensor(audio.size(0)).long()
        return audio, audio_length

    def __getitem__(self, idx):
        item = self.items[idx]
        if not check_speaker_format(item["speaker"]):
            raise ValueError(f"Invalid speaker format at index {idx}: {item}")
        target_audio_filepath = os.path.join(self.audio_base_dir, item["audio_filepath"])
        context_audio_filepath = os.path.join(self.audio_base_dir, item["context_audio_filepath"])
        target_audio, target_audio_length = self.get_wav_from_filepath(target_audio_filepath)
        context_audio, context_audio_length = self.get_wav_from_filepath(context_audio_filepath)
        output_dict = {
            "target_audio_filepath": target_audio_filepath,
            "target_audio": target_audio,
            "target_audio_length": target_audio_length,
            "target_audio_duration": item["duration"],
            "context_audio_filepath": context_audio_filepath,
            "context_audio": context_audio,
            "context_audio_length": context_audio_length,
            "context_audio_duration": item["context_audio_duration"],
            "context_speaker_similarity": item["context_speaker_similarity"],
            "speaker": item["speaker"],
            "text": item["text"],
            "language": item["speaker"].strip().split()[1].split(":")[-1],
        }
        # Extra useful metadata may exist in some manifests, so better to keep them for future usage.
        return self._copy_maybe_extra_metadata(item, output_dict)

    def collate_fn(self, batch):
        max_target_audio_length = max(item["target_audio_length"].item() for item in batch)
        target_audios_padded = [
            torch.nn.functional.pad(item["target_audio"], (0, max_target_audio_length - item["target_audio"].size(0)))
            for item in batch
        ]
        max_context_audio_length = max(item["context_audio_length"].item() for item in batch)
        context_audios_padded = [
            torch.nn.functional.pad(
                item["context_audio"], (0, max_context_audio_length - item["context_audio"].size(0))
            )
            for item in batch
        ]
        output_dict = {
            # target audio
            "target_audio_filepath": [item["target_audio_filepath"] for item in batch],
            "target_audios": torch.stack(target_audios_padded),
            "target_audio_lengths": torch.stack([item["target_audio_length"] for item in batch]),
            "target_audio_durations": [item["target_audio_duration"] for item in batch],
            # context audio
            "context_audio_filepath": [item["context_audio_filepath"] for item in batch],
            "context_audios": torch.stack(context_audios_padded),
            "context_audio_lengths": torch.stack([item["context_audio_length"] for item in batch]),
            "context_audio_durations": [item["context_audio_duration"] for item in batch],
            "context_speaker_similarity": [item["context_speaker_similarity"] for item in batch],
            # metadata
            "speaker": [item["speaker"] for item in batch],
            "text": [item["text"] for item in batch],
            "language": [item["language"] for item in batch],
        }
        # Extra useful metadata may exist in some manifests, so better to keep them for future usage.
        for meta_field in MAYBE_EXTRA_METADATA_IN_MANIFEST:
            if meta_field not in batch[0]:
                continue
            output_dict[meta_field] = [item[meta_field] for item in batch]
        return output_dict

    @staticmethod
    def _copy_maybe_extra_metadata(input_dict: dict, output_dict: dict):
        # Extra useful metadata may exist in some manifests, so better to keep them for future usage.
        for meta_field in MAYBE_EXTRA_METADATA_IN_MANIFEST:
            if meta_field in input_dict:
                output_dict[meta_field] = input_dict[meta_field]
        return output_dict


class CodecExtractor(pl.LightningModule):
    def __init__(self, model_path: str):
        super().__init__()
        self.codec_model = AudioCodecModel.restore_from(restore_path=model_path, strict=False)
        self.codec_model.eval()

    def forward(self, batch):
        with torch.no_grad():
            target_codes, target_codes_lengths = self.codec_model.encode(
                audio=batch["target_audios"], audio_len=batch["target_audio_lengths"]
            )
            context_codes, context_codes_lengths = self.codec_model.encode(
                audio=batch["context_audios"], audio_len=batch["context_audio_lengths"]
            )
        return {
            "target_codes": target_codes.cpu().type(torch.int16),
            "target_codes_lengths": target_codes_lengths,
            "context_codes": context_codes.cpu().type(torch.int16),
            "context_codes_lengths": context_codes_lengths,
        }

    def predict_step(self, batch, batch_idx):
        codes_dict = self(batch)
        target_codes = [
            codes[:, :codes_length]
            for codes, codes_length in zip(codes_dict["target_codes"], codes_dict["target_codes_lengths"])
        ]
        context_codes = [
            codes[:, :codes_length]
            for codes, codes_length in zip(codes_dict["context_codes"], codes_dict["context_codes_lengths"])
        ]
        batch.update(
            {
                "target_codes": target_codes,
                "context_codes": context_codes,
            }
        )
        return batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str)
    parser.add_argument("--audio_base_dir", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--codec_model_name", type=str, default="21fpsCausalDecoder")
    parser.add_argument("--codec_model_path", type=str)
    parser.add_argument("--codec_frame_rate", type=float, default=21.5)
    parser.add_argument("--pad_multiple", type=int, default=1024)
    parser.add_argument("--sample_rate", type=int, default=22050)
    parser.add_argument("--devices", type=int, default=-1)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--shard_size", type=int, default=4096)
    args = parser.parse_args()

    codec_extractor = CodecExtractor(args.codec_model_path)

    dataset = AudioDataset(
        manifest=args.manifest,
        audio_base_dir=args.audio_base_dir,
        sample_rate=args.sample_rate,
        pad_multiple=args.pad_multiple,
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, collate_fn=dataset.collate_fn
    )

    # Note that context_recording would be stored using AudioTarWriter.
    pred_writer = SharPredictionWriter(
        output_dir=args.save_dir,
        codec_model_name=args.codec_model_name,
        audio_base_dir=args.audio_base_dir,
        codec_frame_rate=args.codec_frame_rate,
        fields={
            "recording": "flac",
            f"codes_{args.codec_model_name}": "numpy",
            f"context_codes_{args.codec_model_name}": "numpy",
        },
        shard_size=args.shard_size,
    )

    trainer = Trainer(
        devices=args.devices,
        accelerator="gpu",
        strategy=DDPStrategy(find_unused_parameters=False),
        num_nodes=args.num_nodes,
        logger=False,
    )
    # add writer callback to all gather batched predictions and write into shards.
    trainer.callbacks.append(pred_writer)

    trainer.predict(codec_extractor, dataloaders=dataloader, return_predictions=False)
