import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Union

import torch

from utils import constants

@dataclass
class Token:
    text: str = None
    text_cased: str = None
    s_start: int = None
    s_end: int = None
    t_start: float = None
    t_end: float = None


@dataclass
class BlankToken(Token):
    text: str = constants.BLANK_TOKEN
    text_cased: str = constants.BLANK_TOKEN


@dataclass
class SpaceToken(Token):
    text: str = constants.SPACE_TOKEN
    text_cased: str = constants.SPACE_TOKEN


@dataclass
class Word:
    text: str = None
    s_start: int = None
    s_end: int = None
    t_start: float = None
    t_end: float = None
    tokens: List[Token] = field(default_factory=list)


@dataclass
class Segment:
    text: str = None
    s_start: int = None
    s_end: int = None
    t_start: float = None
    t_end: float = None
    words_and_tokens: List[Union[Word, Token]] = field(default_factory=list)


@dataclass
class Alignment:
    text: str = None
    token_ids_with_blanks: List[int] = field(default_factory=list)
    segments_and_tokens: List[Union[Segment, Token]] = field(default_factory=list)
    saved_output_files: dict = field(default_factory=dict)

    def add_t_start_end(self, viterbri_decoded_batch: List[List[int]], output_timestep_duration: float):
        num_to_first_alignment_appearance = dict()
        num_to_last_alignment_appearance = dict()

        prev_s = -1  # use prev_s to keep track of when the s changes
        for t, s in enumerate(viterbri_decoded_batch):
            if s > prev_s:
                num_to_first_alignment_appearance[s] = t

                if prev_s >= 0:  # dont record prev_s = -1
                    num_to_last_alignment_appearance[prev_s] = t - 1
            prev_s = s
        # add last appearance of the final s
        num_to_last_alignment_appearance[prev_s] = len(viterbri_decoded_batch) - 1

        # update all the t_start and t_end in utt_obj
        for segment_or_token in self.segments_and_tokens:
            if type(segment_or_token) is Segment:
                segment = segment_or_token
                segment.t_start = num_to_first_alignment_appearance[segment.s_start] * output_timestep_duration
                segment.t_end = (num_to_last_alignment_appearance[segment.s_end] + 1) * output_timestep_duration

                for word_or_token in segment.words_and_tokens:
                    if type(word_or_token) is Word:
                        word = word_or_token
                        word.t_start = num_to_first_alignment_appearance[word.s_start] * output_timestep_duration
                        word.t_end = (num_to_last_alignment_appearance[word.s_end] + 1) * output_timestep_duration

                        for token in word.tokens:
                            if token.s_start in num_to_first_alignment_appearance:
                                token.t_start = num_to_first_alignment_appearance[token.s_start] * output_timestep_duration
                            else:
                                token.t_start = -1

                            if token.s_end in num_to_last_alignment_appearance:
                                token.t_end = (
                                    num_to_last_alignment_appearance[token.s_end] + 1
                                ) * output_timestep_duration
                            else:
                                token.t_end = -1
                    else:
                        token = word_or_token
                        if token.s_start in num_to_first_alignment_appearance:
                            token.t_start = num_to_first_alignment_appearance[token.s_start] * output_timestep_duration
                        else:
                            token.t_start = -1

                        if token.s_end in num_to_last_alignment_appearance:
                            token.t_end = (num_to_last_alignment_appearance[token.s_end] + 1) * output_timestep_duration
                        else:
                            token.t_end = -1

            else:
                token = segment_or_token
                if token.s_start in num_to_first_alignment_appearance:
                    token.t_start = num_to_first_alignment_appearance[token.s_start] * output_timestep_duration
                else:
                    token.t_start = -1

                if token.s_end in num_to_last_alignment_appearance:
                    token.t_end = (num_to_last_alignment_appearance[token.s_end] + 1) * output_timestep_duration
                else:
                    token.t_end = -1
        return

@dataclass
class Utterance:
    utt_id: str = None   
    audio_filepath: str = None
    text: Alignment = field(default_factory=Alignment)
    pred_text: Alignment = field(default_factory=Alignment)

    def _set_utt_id(self, audio_filepath_parts_in_utt_id: str):
        fp_parts = Path(self.audio_filepath).parts[-audio_filepath_parts_in_utt_id:]
        self.utt_id = Path("_".join(fp_parts)).stem
        self.utt_id = self.utt_id.replace(" ", "-")  # replace any spaces in the filepath with dashes

    @staticmethod
    def get_utterance(audio_filepath: str, text: str = None, pred_text: str = None, audio_filepath_parts_in_utt_id: int = 1):
        utt = Utterance(audio_filepath=audio_filepath)
        utt._set_utt_id(audio_filepath_parts_in_utt_id)
        utt.text.text = text
        utt.pred_text.text = text
        return utt


@dataclass
class TokensBatch:
    y_list: List[List] = field(default_factory=list)
    y: torch.Tensor = field(default_factory=lambda: torch.tensor([]))
    U_list: List[int] = field(default_factory=list)
    U_max: int = None
    U: torch.Tensor = field(default_factory=lambda: torch.tensor([]))

    def to_tensor(self, V: int, B: int):
        self.U_max = max(self.U_list)
        self.U = torch.tensor(self.U_list)
        self.y = V * torch.ones((B, self.U_max), dtype=torch.int64)
        
        for b, y_utt in enumerate(self.y_list):
            U_utt = self.U[b]
            self.y[b, :U_utt] = torch.tensor(y_utt)


@dataclass
class Batch:
    B: int = None
    manifest_lines: List[Dict] = field(default_factory=list)
    audio_filepaths: List[str] = field(default_factory=list)
    log_probs_list: List[List] = field(default_factory=list)
    log_probs: torch.Tensor = field(default_factory=lambda: torch.tensor([]))
    T_list: List[List] = field(default_factory=list)
    T_max: int = None
    T: torch.Tensor = field(default_factory=lambda: torch.tensor([]))
    pred_texts: List[str] = field(default_factory=list)
    utterances: List[Utterance] = field(default_factory=list)
    texts_batch: TokensBatch = field(default_factory=TokensBatch)
    pred_texts_batch: TokensBatch = field(default_factory=TokensBatch)
    output_timestep_duration: float = None

    @staticmethod
    def _read_manifest(manifest_filepath):
        with open(manifest_filepath, "rt", encoding="utf8") as manifest:
            for line in manifest:
                data = json.loads(line)
                if "text" in data:
                    # remove any BOM, any duplicated spaces, convert any
                    # newline chars to spaces
                    data["text"] = data["text"].replace("\ufeff", "")
                    data["text"] = " ".join(data["text"].split())

                    # Replace any horizontal ellipses with 3 separate periods.
                    # The tokenizer will do this anyway. But making this replacement
                    # now helps avoid errors when restoring punctuation when saving
                    # the output files
                    data["text"] = data["text"].replace("\u2026", "...")
                yield data
    
    @staticmethod
    def chunk_manifest(manifest_filepath: str, batch_size: int):
        manifest_chunk = []
        for idx, data_entry in enumerate(Batch._read_manifest(manifest_filepath), 1):
            manifest_chunk.append(data_entry)
            if idx % batch_size == 0:
                yield manifest_chunk
                manifest_chunk = []
        if len(manifest_chunk) > 0:
            yield manifest_chunk
    
    @staticmethod
    def get_batch(manifest_lines_batch):
        batch = Batch(manifest_lines=manifest_lines_batch)
        batch._set_audio_filepaths()
        batch._set_B()
        return batch
    
    def _set_audio_filepaths(self):
        self.audio_filepaths = [data["audio_filepath"] for data in self.manifest_lines]
    
    def _set_B(self):
        self.B = len(self.manifest_lines)
    
    def set_utterances(self, audio_filepath_parts_in_utt_id: int, align_using_text: bool = True):
        for data, audio_filepath, pred_text in zip(self.manifest_lines, self.audio_filepaths, self.pred_texts):
            text = data['text'] if align_using_text else None
            pred_text = " ".join(pred_text.split())
            
            utt = Utterance.get_utterance(audio_filepath=audio_filepath,
                                          text = text,
                                          pred_text = pred_text,
                                          audio_filepath_parts_in_utt_id = audio_filepath_parts_in_utt_id)

            self.utterances.append(utt)
    
    def to_tensor(self, V: int):
        for utterance in self.utterances:
            if utterance.text:
                self.texts_batch.y_list.append(utterance.text.token_ids_with_blanks)
                self.texts_batch.U_list.append(len(utterance.text.token_ids_with_blanks))
            if utterance.pred_text:
                self.pred_texts_batch.y_list.append(utterance.pred_text.token_ids_with_blanks)
                self.pred_texts_batch.U_list.append(len(utterance.pred_text.token_ids_with_blanks))

        self.T_max = max(self.T_list)
        self.T = torch.tensor(self.T_list)
        self.log_probs = constants.V_NEGATIVE_NUM * torch.ones((self.B, self.T_max, V))

        for b, log_probs_utt in enumerate(self.log_probs_list):
            t = log_probs_utt.shape[0]
            self.log_probs[b, :t, :] = log_probs_utt
        
        if len(self.texts_batch.U_list) > 0:
            self.texts_batch.to_tensor(V, self.B)
        
        if len(self.pred_texts_batch.U_list) > 0:
            self.pred_texts_batch.to_tensor(V, self.B)