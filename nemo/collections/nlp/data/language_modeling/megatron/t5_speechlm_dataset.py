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

import copy
import json
import random
from itertools import filterfalse
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch
from encodec import EncodecModel
from tqdm.auto import tqdm

from nemo.collections.asr.parts.preprocessing.features import WaveformFeaturizer
from nemo.collections.asr.parts.preprocessing.segment import AudioSegment
from nemo.collections.nlp.data.language_modeling.megatron.base_prompt_learning_dataset import BasePromptLearningDataset
from nemo.collections.nlp.models.language_modeling.megatron_t5_model import T5Sentinel
from nemo.collections.nlp.modules.common import VirtualPromptSource
from nemo.collections.nlp.modules.common.megatron.utils import build_position_ids
from nemo.collections.tts.parts.utils.helpers import get_mask_from_lengths
from nemo.collections.tts.parts.utils.tts_dataset_utils import general_padding, get_base_dir
from nemo.utils import logging

__all__ = ['T5SpeechLMDataset']


def pad_text_to_speech_dims(text_tensor, pad_id):
    token_len = text_tensor.shape[0]
    empty_padding = torch.ones((7, token_len), dtype=text_tensor.dtype, device=text_tensor.device) * pad_id
    return torch.cat((text_tensor.unsqueeze(0), empty_padding), dim=0)

class GPTSpeechLMDataset(BasePromptLearningDataset):
    def __getitem__(self, idx):
        pass


class T5SpeechLMDataset(BasePromptLearningDataset):
    """
    The dataset class for prompt-tuning or p-tuning pretrained T5 SpeechLM models.
    """

    def __init__(
        self,
        datasets,
        tokenizer,
        virtual_prompt_source: VirtualPromptSource,
        task_templates: dict,
        pseudo_tokens,
        pad_token_id: str,
        max_seq_length: int,
        sample_rate: int,
        min_seq_length: int = 1,
        add_bos: bool = False,
        add_eos: bool = True,
        for_train: bool = True,
        decoder_starts_with_pad: bool = False,
        add_eos_to_decoder_output: bool = True,
        add_sentinel_to_input: bool = True,
        ul2_prompt_token: str = None,
        segment_max_duration: Optional[int] = None,
        trim: bool = False,
        trim_ref: Optional[float] = None,
        trim_top_db: Optional[int] = None,
        trim_frame_length: Optional[int] = None,
        trim_hop_length: Optional[int] = None,
        pad_multiple: int = 1,
        pitch_augment: bool = False,
        sup_data_path: Optional[Union[Path, str]] = None,
        speech_offset: Optional[int] = None,
        train_task: Optional[str] = None,
        seq_pattern: Optional[str] = "parallel",
        **kwargs,
    ):
        """
        Only speech parameters are explained here.
        segment_max_duration: Optional[int] = None, - Speech max segment duration
        trim: bool = False, - speech parameter
        trim_ref: Optional[float] = None, - speech parameter
        trim_top_db: Optional[int] = None, - speech parameter
        trim_frame_length: Optional[int] = None, - speech parameter
        trim_hop_length: Optional[int] = None, - speech parameter
        pad_multiple: int = 1, - speech parameter
        pitch_augment: bool = False, - speech parameter
        sup_data_path: Optional[Union[Path, str]] = None, - Supplementary folder path where codecs are stored.
        speech_offset: Optional[int] = None, - if speech tokens then add this offset to the token indices to distinguish between text and speech tokens.
        **kwargs,
        """
        # These two variables need to be set before calling super().__init__() because the parent class calls `load_data()` which requires these attributes.
        self.decoder_starts_with_pad = decoder_starts_with_pad
        self.add_eos_to_decoder_output = add_eos_to_decoder_output
        self.add_sentinel_to_input = add_sentinel_to_input
        self.ul2_prompt_token = ul2_prompt_token
        # Speech related variables
        self.encodec_model = EncodecModel.encodec_model_24khz()
        self.encodec_model.set_target_bandwidth(6.0)
        self.base_data_dir = None
        self.segment_max_duration = segment_max_duration
        self.sample_rate = sample_rate
        self.featurizer = WaveformFeaturizer(sample_rate=self.sample_rate)
        self.pad_multiple = pad_multiple
        self.pitch_augment = pitch_augment
        self.trim = trim
        self.trim_ref = trim_ref if trim_ref is not None else np.max
        self.trim_top_db = trim_top_db if trim_top_db is not None else 60
        self.trim_frame_length = trim_frame_length if trim_frame_length is not None else 2048
        self.trim_hop_length = trim_hop_length if trim_hop_length is not None else 512
        self.speech_offset = speech_offset if speech_offset is not None else 3
        self.seq_pattern = seq_pattern

        # Initialize sup_data_path, sup_data_types and run preprocessing methods for every supplementary data type
        if sup_data_path is not None:
            Path(sup_data_path).mkdir(parents=True, exist_ok=True)
            self.sup_data_path = sup_data_path

        self.codec_folder = kwargs.pop('codec_folder', None)
        self.train_task = train_task
        if self.codec_folder is None:
            self.codec_folder = Path(self.sup_data_path) / "codec"
        elif isinstance(self.codec_folder, str):
            self.codec_folder = Path(self.codec_folder)

        self.codec_folder.mkdir(exist_ok=True, parents=True)

        super().__init__(
            datasets=datasets,
            tokenizer=tokenizer,
            virtual_prompt_source=virtual_prompt_source,
            task_templates=task_templates,
            pseudo_tokens=pseudo_tokens,
            pad_token_id=pad_token_id,
            max_seq_length=max_seq_length,
            min_seq_length=min_seq_length,
            add_bos=add_bos,
            add_eos=add_eos,
            for_train=for_train,
        )

    def load_data(self, dataset):
        """
        Loads a dataset by filling in the task templates specified in the config file
        with the information from each training/inference example. Converts all input
        text into token ids. Also replaces the <|VIRTUAL_PROMPT_#|> placeholders in
        the task templates with the actual virtual prompt token ids.

        params:
            dataset: A list of json objects or a dictionary objects each
                     containing the information needed for a training example
        """
        copy_dataset = list(dataset)
        audio_filelist = []
        # This loop is needed to calculate self.base_data_dir.
        for json_line in copy_dataset:
            if type(json_line) == dict:
                doc = json_line
            else:
                doc = json.loads(json_line)
            taskname = doc["taskname"]
            prompt_template_fields = self.task_templates[taskname]["prompt_template_fields"]

            for p in prompt_template_fields:
                if f"{p}_type" in doc and doc[f"{p}_type"] == "SPEECH":
                    audio_filelist.append(doc[p])
        self.base_data_dir = get_base_dir(audio_filelist)

        skipped = 0
        tts = 0
        i = 0
        print(f"copy_dataset len === {len(copy_dataset)}")
        for json_line in tqdm(copy_dataset):
            i += 1

            # Read example dict or load the information for a single example from .json file
            if type(json_line) == dict:
                doc = json_line
            else:
                doc = json.loads(json_line)

            question_in_manifest = doc['question']

            if "Text to speech this" in question_in_manifest:
                tts += 1
                if self.train_task != 'tts':
                    continue
            elif "Next token prediction" in question_in_manifest:
                tts += 1
                pass
            else:
                if self.train_task == 'tts':
                    continue

            if doc["context_type"] == "SPEECH":
                assert "context_duration" in doc, f"context_duration key not in document {doc}"
                approx_context_len = min(doc["context_duration"] * 76 * 0.3, 400)
            approx_question_len = len(doc["question"].split(' ')) + 3

            if doc["answer_type"] == "SPEECH":
                assert "answer_duration" in doc, f"answer_duration key not in document {doc}"
                approx_answer_len = doc["answer_duration"] * 76
                if self.seq_pattern == "delay_parallel":
                    # In delay parallel, there is padding so add 8 frames
                    approx_answer_len = approx_answer_len + 8
            else:
                approx_answer_len = len(doc["answer"].split(' ')) + 3

            if (
                approx_context_len + approx_question_len < self.max_seq_length
                and approx_answer_len < self.max_seq_length
            ):
                self.examples.append(doc)
            else:
                print(f"skipped for {approx_context_len + approx_question_len} {approx_answer_len} len")
                skipped += 1

        print(f"After Process len(self.examples) {len(self.examples)} {tts}")
        logging.info(f'Skipped {skipped} sentences, sequence length too short or too long even after truncation')

    def __getitem__(self, idx):
        doc = self.examples[idx]
        taskname = doc["taskname"]
        prompt_template = self.task_templates[taskname]["prompt_template"]
        prompt_template_fields = self.task_templates[taskname]["prompt_template_fields"]
        total_virtual_tokens = self.task_templates[taskname]["total_virtual_tokens"]
        virtual_token_splits = self.task_templates[taskname]["virtual_token_splits"]
        truncation_field = self.task_templates[taskname]['truncate_field']
        answer_field = self.task_templates[taskname]["answer_field"]

        input_example = prompt_template

        self._input_sanity_checks(
            total_virtual_tokens=total_virtual_tokens,
            virtual_token_splits=virtual_token_splits,
            prompt_template=prompt_template,
            prompt_template_fields=prompt_template_fields,
            truncation_field=truncation_field,
            answer_field=answer_field,
            doc=doc,
        )
        question_in_manifest = doc['question']

        # Format the input example according to the template
        # Get context, question and answer codes in a dict.
        input_dict = self._insert_data_in_template(input_example, prompt_template_fields, doc, answer_field)
        context_tokens = input_dict['context']
        question_tokens = input_dict['question']

        # Logic to prune context
        # In case of TTS task, the entire reference speech is not required, so we randomly select a portion
        # of the reference audio.
        # In case of Next token prediction, We want context[:T] to go in the encoder and context[T+1:] to be
        # predicted by the decoder.
        start_token_index = 0
        end_token_index = -1
        if "Text to speech this" in question_in_manifest:
            total_context_len = context_tokens[0].size()[1]
            reduced_len = min(
                400,
                int(total_context_len * 0.2)
                if total_context_len > 600
                else int(total_context_len * random.uniform(0.2, 0.5)),
            )
            start_token_index = random.randint(
                0, total_context_len - reduced_len
            )  # start index can be greater than 440
            context_tokens[0] = context_tokens[0][
                :, start_token_index : min(start_token_index + 440, start_token_index + reduced_len)
            ]
        elif "Next token prediction" in question_in_manifest:
            total_context_len = context_tokens[0].size()[1]
            end_token_index = int(total_context_len * random.uniform(0.01, 0.2))
            context_tokens[0] = context_tokens[0][:, :end_token_index]

        # Get virtual tokens
        virtual_tokens = self._insert_virtual_token_placeholders(input_example.split(' ')[0], virtual_token_splits)

        # a trick to align with the data format in t5 pretraining
        # new
        virtual_tokens = self.tokenizer.text_to_ids(virtual_tokens)
        if self.add_sentinel_to_input:
            question_tokens = question_tokens + self.tokenizer.text_to_ids(T5Sentinel.FIRST.value)

        # Add BOS/EOS to the input of encoder if desired, adds EOS by default
        if self.ul2_prompt_token is not None:
            ul2_prompt_token_id = self.tokenizer.text_to_ids(self.ul2_prompt_token)
            assert len(ul2_prompt_token_id) == 1
            context_tokens = ul2_prompt_token_id + context_tokens
        if self.add_bos:
            context_tokens = [self.tokenizer.bos_id] + context_tokens
        if self.add_eos:
            question_tokens = question_tokens + [self.tokenizer.eos_id]

        # Try to truncate input text to fit into the max sequence length
        if self._get_len(context_tokens, question_tokens, virtual_tokens) > self.max_seq_length:
            context_tokens, question_tokens, virtual_tokens = self._truncate_input_speech(
                context_tokens, question_tokens, virtual_tokens
            )

        virtual_tokens, virtual_tokens_len = self.list_to_tensor(virtual_tokens)
        context_tokens, context_tokens_len = self.list_to_tensor(context_tokens)
        question_tokens, question_tokens_len = self.list_to_tensor(question_tokens)

        if doc["question_type"] != "SPEECH" and doc["context_type"] == "SPEECH":
            question_tokens = pad_text_to_speech_dims(question_tokens, self.tokenizer.pad_id)
        if doc["context_type"] != "SPEECH" and doc["question_type"] == "SPEECH":
            context_tokens = pad_text_to_speech_dims(context_tokens, self.tokenizer.pad_id)
        context_and_question_tokens = torch.cat([context_tokens, question_tokens], dim=1)

        # get answer ids
        if answer_field in doc.keys():  # training and validation
            answer_ids = self._get_tokens(doc, answer_field, doc[answer_field])
            if end_token_index > -1:
                answer_ids[0] = answer_ids[0][:, end_token_index:]

            if self.decoder_starts_with_pad:
                answer_text_ids = [self.tokenizer.pad_id]
            else:
                answer_text_ids = [self.tokenizer.bos_id]
            # a trick to align with the data format in t5 pretraining
            # if self.add_sentinel_to_input:
            #     answer_text_ids += self.tokenizer.text_to_ids(T5Sentinel.FIRST.value)
            answer_text_ids += answer_ids

            if self.add_eos_to_decoder_output:
                answer_text_ids += [self.tokenizer.eos_id]
            else:
                answer_text_ids += self.tokenizer.text_to_ids(T5Sentinel.END.value)

        # Skip example if the final length doesn't fit length requirements even after truncation
        if (
            self.min_seq_length
            <= self._get_element_len(context_and_question_tokens) + self._get_element_len(virtual_tokens)
            <= self.max_seq_length
            and self.min_seq_length <= self._get_element_len(answer_text_ids) <= self.max_seq_length
        ):
            if self.virtual_prompt_source == VirtualPromptSource.PROMPT_ENCODER:
                taskname_id = self.tokenizer.text_to_ids(taskname)
            elif (
                self.virtual_prompt_source == VirtualPromptSource.NO_PROMPT
            ):  # TODO (@adithyare) this class and GPTPromptLearningDataset should be merged.
                taskname_id = -1
            else:
                raise ValueError("Invalid virtual prompt source specified")

            dec_input = None
            dec_labels = None

            if answer_field in doc.keys():  # training and validation
                dec_input = answer_text_ids[:-1]
                dec_labels = answer_text_ids[1:]

            dec_input, dec_input_len = self.list_to_tensor(dec_input, True)
            dec_labels, dec_labels_len = self.list_to_tensor(dec_labels, True)
            is_speech = True if doc["answer_type"] == "SPEECH" else False
            if is_speech:
                assert dec_input.dim() == 2 and dec_labels.dim() == 2
                if self.seq_pattern == "delay_parallel":
                    num_codebooks = dec_input.shape[0]
                    dec_input_padded = torch.cat(
                        [
                            torch.zeros_like(dec_input[:, 0:num_codebooks]),
                            dec_input,
                            torch.zeros_like(dec_input[:, 0:num_codebooks]),
                        ],
                        dim=1,
                    )
                    dec_labels_padded = torch.cat(
                        [
                            torch.zeros_like(dec_labels[:, 0:num_codebooks]),
                            dec_labels,
                            torch.zeros_like(dec_labels[:, 0:num_codebooks]),
                        ],
                        dim=1,
                    )
                    dec_input_new = []
                    dec_labels_new = []
                    for _c in range(8):
                        st = num_codebooks - _c
                        et_decoder_input = dec_input_padded.shape[1] - _c
                        et_decoder_labels = dec_labels_padded.shape[1] - _c
                        dec_input_new.append(dec_input_padded[_c, st:et_decoder_input])
                        dec_labels_new.append(dec_labels_padded[_c, st:et_decoder_labels])
                    dec_input = torch.stack(dec_input_new, dim=0)
                    dec_labels = torch.stack(dec_labels_new, dim=0)
                    dec_input_len = torch.tensor(dec_input.shape[1]).long()
                    dec_labels_len = torch.tensor(dec_labels.shape[1]).long()

            return (
                taskname_id,
                virtual_tokens,
                virtual_tokens_len,
                context_and_question_tokens,
                context_tokens_len + question_tokens_len,
                dec_input,
                dec_input_len,
                dec_labels,
                dec_labels_len,
                is_speech,
            )

    def _truncate_input_speech(self, context_tokens, question_tokens, virtual_tokens):
        total_len = self._get_len(context_tokens, question_tokens, virtual_tokens)
        context_len = self._get_element_len(context_tokens)
        truncation_length = total_len - self.max_seq_length + 1
        context_tokens[0] = context_tokens[0][:, min(truncation_length, context_len) :]
        return context_tokens, question_tokens, virtual_tokens

    def list_to_tensor(self, element, fill=False):
        """
        Convert list to tensor. The list might contain integers, 2D-tensors (speech tokens) and combination of two.
        If all of them are ints, simply convert to tensor
        If combination of 2D-tensor and ints. Convert int to the dimension of the tensor.
        example: [2, 4, 5] -> torch.tensor([2, 4, 5])
        example: [2, torch.tensor([[4, 5, 6], [6, 7, 8]])] -> torch.tensor( [[-1, 4, 5, 6], [2, 6, 7, 8]] )
        """
        ret, ln = None, None
        if element is None:
            return ret, ln

        max_len = max([1 if isinstance(item, int) else len(item) for item in element])
        if max_len == 1:
            ret = torch.as_tensor(element).long()
            ln = torch.tensor(ret.size()[0]).long()
        else:
            ret = []
            for e in element:
                if isinstance(e, int):
                    tmp = torch.full((8, 1), e if fill else -1)
                    tmp[7] = e
                else:
                    tmp = e
                ret.append(tmp)
            ret = torch.cat(ret, dim=1)
            ln = torch.tensor(ret.size()[1]).long()
        return ret, ln

    def _get_text_tokens(self, text):
        input_ids = self.tokenizer.text_to_ids(text)
        return input_ids

    def _pad_wav_to_multiple(self, wav):
        if self.pad_multiple > 1:
            if wav.shape[0] % self.pad_multiple != 0:
                wav = torch.cat(
                    [wav, torch.zeros(self.pad_multiple - wav.shape[0] % self.pad_multiple, dtype=torch.float)]
                )
        return wav

    def _get_element_len(self, element):
        length = 0
        if isinstance(element, list):
            for e in element:
                if isinstance(e, int):
                    length += 1
                else:
                    if e.dim() > 1:
                        length += e.size()[1]
                    else:
                        length += e.size()[0]
        else:
            if element.dim() > 1:
                length += element.size()[1]
            else:
                length += element.size()[0]
        return length

    def _get_len(self, context_tokens, question_tokens, virtual_tokens):
        length = 0
        length += self._get_element_len(context_tokens)
        length += self._get_element_len(question_tokens)
        length += self._get_element_len(virtual_tokens)
        return length

    def _load_audio(self, audio_filepath, dur=-1):
        if self.segment_max_duration is not None and dur > 0 and dur > self.segment_max_duration:
            # this case has been added for segmenting audio for speaker verification task of SSLDisentangler
            n_segments = int(self.segment_max_duration * self.sample_rate)
            features = AudioSegment.segment_from_file(
                audio_filepath, target_sr=self.sample_rate, n_segments=n_segments, trim=self.trim
            )

            features = torch.tensor(features.samples)
            if self.pad_multiple > 1:
                features = self._pad_wav_to_multiple(features)
            audio, audio_length = features, torch.tensor(features.shape[0]).long()
        else:
            features = self.featurizer.process(
                audio_filepath,
                trim=self.trim,
                trim_ref=self.trim_ref,
                trim_top_db=self.trim_top_db,
                trim_frame_length=self.trim_frame_length,
                trim_hop_length=self.trim_hop_length,
            )

            if self.pad_multiple > 1:
                features = self._pad_wav_to_multiple(features)

            audio, audio_length = features, torch.tensor(features.shape[0]).long()

        return audio, audio_length

    def convert_audio(self, audio, sample_rate, target_sample_rate, target_channels):
        if audio.dim() == 1:
            audio = audio.unsqueeze(0).unsqueeze(0)
        assert audio.shape[1] in [1, 2], "Audio must be mono or stereo."
        # assert sample_rate == target_sample_rate, "sample rate of FastPitch and Encodec model has to be same"
        if target_channels == 2:
            *shape, _, length = audio.shape
            audio = audio.expand(*shape, target_channels, length)
        return audio

    def get_codec(self, audio):
        wav1 = self.convert_audio(audio, self.sample_rate, self.encodec_model.sample_rate, self.encodec_model.channels)
        encoded_frames = self.encodec_model.encode(wav1)
        codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)
        return codes.squeeze(0)

    def get_quantizer_codebook(self, reference_codec, reference_codec_length):
        out = torch.zeros((1, 128, reference_codec_length.item()))
        for i in range(reference_codec.size()[0]):
            out += self.encodec_model.quantizer.vq.layers[i].decode(reference_codec[i, :].unsqueeze(0))
        return out.squeeze(0)

    def _get_speech_tokens(self, audio_filepath, dur=-1):
        # Let's keep audio name and all internal directories in rel_audio_path_as_text_id to avoid any collisions
        rel_audio_path = Path(audio_filepath).relative_to(self.base_data_dir).with_suffix("")
        rel_audio_path_as_text_id = str(rel_audio_path).replace("/", "_")

        # Load audio features
        audio, audio_length = self._load_audio(audio_filepath, dur)

        # Convert to codes
        codec_codes, codec_codes_length = None, None  # Codes
        codec_path = self.codec_folder / f"{rel_audio_path_as_text_id}.pt"

        if codec_path.exists():
            try:
                codec_codes = torch.load(codec_path).long()
            except Exception as e:
                print(f"[ERROR IN LOADING {codec_path}] e")
                codec_codes = self.get_codec(audio).long()
                torch.save(codec_codes, codec_path)
        else:
            codec_codes = self.get_codec(audio).long()
            torch.save(codec_codes, codec_path)

        codec_codes_length = torch.tensor(codec_codes.shape[1]).long()

        # Convert codes to codes corresponding to megatron embedding layer
        codec_codes[0] = (codec_codes[0] + self.speech_offset).long()

        return codec_codes

    def _get_tokens(self, doc, field, field_data):
        if f"{field}_type" not in doc.keys():
            field_tokens = self._get_text_tokens(field_data.strip(" "))  # list of ids
        elif doc[f"{field}_type"] == 'TEXT':
            field_tokens = self._get_text_tokens(field_data.strip(" "))  # list of ids
        elif doc[f"{field}_type"] == 'SPEECH':
            dur = -1
            if f"{field}_duration" in doc:
                dur = doc[f"{field}_duration"]
            field_tokens = self._get_speech_tokens(field_data, dur)  # list of ids
            if not isinstance(field_tokens, list):
                field_tokens = [field_tokens]
        else:
            raise Exception(f"{field}_type not recognized")
        return field_tokens

    def _insert_data_in_template(self, input_example, prompt_template_fields, doc, answer_field):
        """ Format the input example according to the template """
        out_dict = {}
        for field in prompt_template_fields:
            # discard the last one, {label} / {answer}
            # Or if some fields from the template aren't present, e.g. {answer} during inference
            # just remove that field from the template, leaving the space blank
            if field == answer_field or field not in doc.keys():
                continue
                #  out_dict[field] = ""

            elif field in doc.keys():
                field_data = doc[field]
                if f"{field}_type" not in doc.keys():
                    doc[f"{field}_type"] = "TEXT"
                    raise Exception(f"{field}_type does not exist in doc")
                else:
                    out_dict[field] = self._get_tokens(doc, field, field_data)
        return out_dict

    def get_position_ids(self, virtual_token, context_and_qquestion):
        enc_input = []
        enc_input.append(virtual_token)
        if context_and_qquestion.dim() > 2:
            enc_input.append(context_and_qquestion[:, 0, :])
        else:
            enc_input.append(context_and_qquestion)

        enc_input = torch.cat(enc_input, dim=1)

        return build_position_ids(enc_input).contiguous()

    def collate_fn(self, batch):
        """ Prepares enc_input, dec_input, labels, loss_mask, enc_mask, dec_mask, position_ids, taskname_ids for global batch """

        data_dict = self.pad_batch_and_build_loss_mask(batch)

        position_ids = self.get_position_ids(data_dict['virtual_tokens'], data_dict['context_and_question_tokens'])

        return (
            data_dict['virtual_tokens'],
            data_dict['context_and_question_tokens'],
            data_dict['enc_mask'],
            data_dict['dec_input'],
            data_dict['dec_input_mask'],
            data_dict['dec_labels'],
            data_dict['dec_labels_mask'],
            position_ids,
            data_dict['taskname_id'],
            data_dict['speech_mask'],
        )

    def pad_batch_and_build_loss_mask(self, batch):
        """ Pad enc_input, dec_input, labels in batch to max batch length while building loss_mask, enc_mask, and dec_mask """
        (
            taskname_ids,
            _,
            virtual_tokens_len,
            _,
            context_and_question_tokens_len,
            _,
            dec_input_len,
            _,
            dec_labels_len,
            _,
        ) = zip(*batch)

        taskname_ids = self.pad_taskname_ids(taskname_ids)

        max_virtual_tokens_len = max(virtual_tokens_len).item() if virtual_tokens_len is not None else 0
        if isinstance(virtual_tokens_len, tuple):
            virtual_tokens_len = torch.stack(virtual_tokens_len)
        virtual_mask = get_mask_from_lengths(virtual_tokens_len)

        max_context_and_question_tokens_len = (
            max(context_and_question_tokens_len).item() if context_and_question_tokens_len is not None else 0
        )
        if isinstance(context_and_question_tokens_len, tuple):
            context_and_question_tokens_len = torch.stack(context_and_question_tokens_len)
        context_and_question_mask = get_mask_from_lengths(context_and_question_tokens_len)

        max_dec_input_len = max(dec_input_len).item() if dec_input_len is not None else 0
        max_dec_labels_len = max(dec_labels_len).item() if dec_labels_len is not None else 0
        enc_mask = torch.cat([virtual_mask, context_and_question_mask], dim=1)

        (
            virtual_tokens_list,
            context_question_tokens_list,
            dec_input_list,
            dec_input_mask_list,
            dec_labels_list,
            dec_labels_mask_list,
            speech_mask_list,
        ) = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )

        for i, sample_tuple in enumerate(batch):
            (
                _,
                virtual_token,
                virtual_token_len,
                context_and_question_token,
                context_and_question_token_len,
                dec_input,
                dec_input_len,
                dec_label,
                dec_label_len,
                is_speech,
            ) = sample_tuple

            virtual_tokens_list.append(
                general_padding(
                    virtual_token, virtual_token_len.item(), max_virtual_tokens_len, pad_value=self.tokenizer.pad_id
                )
            )

            context_tokens_padded = general_padding(
                context_and_question_token,
                context_and_question_token_len.item(),
                max_context_and_question_tokens_len,
                pad_value=self.tokenizer.pad_id,
            )
            if len(context_tokens_padded.shape) < 2:
                context_tokens_padded = pad_text_to_speech_dims(context_tokens_padded, self.tokenizer.pad_id)
            context_question_tokens_list.append(context_tokens_padded)

            if max_dec_input_len > 0:
                dec_input_padded = general_padding(
                    dec_input, dec_input_len.item(), max_dec_input_len, pad_value=self.tokenizer.pad_id
                )
                if len(dec_input_padded.shape) < 2:
                    dec_input_padded = pad_text_to_speech_dims(dec_input_padded, self.tokenizer.pad_id)
                dec_input_list.append(dec_input_padded)
                dec_mask = (
                    torch.as_tensor(([1] * dec_input_len) + ([0] * (max_dec_input_len - dec_input_len)))
                    .long()
                    .contiguous()
                )
                dec_input_mask_list.append(dec_mask)
                speech_mask = dec_mask if is_speech else torch.zeros(dec_mask.shape)
                speech_mask_list.append(speech_mask)

            if max_dec_labels_len > 0:
                loss_mask = (
                    torch.as_tensor(([1] * dec_label_len) + ([0] * (max_dec_labels_len - dec_label_len)))
                    .long()
                    .contiguous()
                )
                dec_label_padded = general_padding(
                    dec_label, dec_label_len.item(), max_dec_labels_len, pad_value=self.tokenizer.pad_id
                )
                if len(dec_label_padded.shape) < 2:
                    dec_label_padded = pad_text_to_speech_dims(dec_label_padded, self.tokenizer.pad_id)
                dec_labels_list.append(dec_label_padded)
                dec_labels_mask_list.append(loss_mask)

        data_dict = {
            "taskname_id": taskname_ids,
            "virtual_tokens": torch.stack(virtual_tokens_list),
            "context_and_question_tokens": torch.stack(context_question_tokens_list),
            "enc_mask": enc_mask,
            "dec_input": torch.stack(dec_input_list) if len(dec_input_list) > 0 else None,
            "dec_input_mask": torch.stack(dec_input_mask_list) if len(dec_input_mask_list) > 0 else None,
            "dec_labels": torch.stack(dec_labels_list) if len(dec_labels_list) > 0 else None,
            "dec_labels_mask": torch.stack(dec_labels_mask_list) if len(dec_labels_mask_list) > 0 else None,
            "speech_mask": torch.stack(speech_mask_list) if len(speech_mask_list) > 0 else None,
        }

        return data_dict
