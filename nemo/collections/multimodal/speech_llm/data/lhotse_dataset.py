import logging
import random

import torch.utils.data
from lhotse import CutSet
from lhotse.dataset import AudioSamples
from lhotse.dataset.collation import _read_features
from lhotse.dataset.collation import collate_vectors as collate_vectors_lhotse

from nemo.collections.multimodal.speech_llm.parts.utils.data_utils import (
    TextProcessing,
    build_loss_mask,
    ceil_to_nearest,
)


def collate_vectors(items, max_length: int, padding_value):
    vectors = collate_vectors_lhotse(items, padding_value=padding_value)
    if max_length > vectors.size(1):
        vectors = torch.cat(
            [vectors, padding_value * torch.ones(vectors.size(0), max_length - vectors.size(1), dtype=vectors.dtype)],
            dim=1,
        )
    if items[0].shape[0] < 1:
        vectors = vectors.long()
    return vectors


# TODO: the changes in this file needed to be moved out as a derived class
class LhotseAudioQuestionAnswerDataset(torch.utils.data.Dataset):
    """
    This dataset is based on Lhotse ASR dataset from ``audio_to_text_lhotse.py``
    and ``TarredAudioQuestionAnswerDataset`` from ``audio_text_qa_dataset.py``.

    Unlike native NeMo datasets, Lhotse dataset defines only the mapping from
    a CutSet (meta-data) to a mini-batch with PyTorch tensors.
    Specifically, it performs tokenization, I/O, augmentation, and feature extraction (if any).
    Managing data, sampling, de-duplication across workers/nodes etc. is all handled
    by Lhotse samplers instead.

    Args:
        text_processor: TextProcessing object
        default_context: Default question to use if no question is provided
        tokens_to_generate: Number of tokens to generate during inference
        pad_to_max_length: Whether to pad the input to the max sequence length. If False, will pad to the max length of the current batch.
        max_seq_length: Maximum sequence length for each dataset examples. Examples will either be truncated to fit this length or dropped if they cannot be truncated.
        context_key: Key to use for the context in your JSONL file
        default_context_key: Key to use for the default context in lhotse yaml
    """

    def __init__(
        self,
        text_processor: TextProcessing,
        default_context: str,
        tokens_to_generate: int,
        pad_to_max_length: bool,
        max_seq_length: int,
        context_key: str = "context",
        default_context_key: str = "default_context",
        vocab_sizes: list[int] = [-1],
        decoder_reduction_factor: int = 1,
        speech_pad_id: int = 1001,
        speech_unk_id: int = 1002,
        speech_bos_id: int = 1003,
        speech_eos_id: int = 1004,
        filter_by_source_target_text_ratio: bool = False,
        source_target_text_ratio_limit: float = 1.0,
        sample_rate: int = 22050,
        t5_style: bool = False,
    ):
        super().__init__()
        self.text_processor = text_processor
        self.load_audio = AudioSamples(fault_tolerant=True)
        self.tokens_to_generate = tokens_to_generate
        self.pad_to_max_length = pad_to_max_length
        self.max_seq_length = max_seq_length

        self.default_context = default_context
        self.context_key = context_key
        self.default_context_key = default_context_key

        if len(vocab_sizes) == 1 and vocab_sizes[0] <= 0:
            vocab_sizes = [self.text_processor.tokenizer.vocab_size]
        self.vocab_sizes = list(vocab_sizes)
        self.n_speech_codebooks = len(self.vocab_sizes) - 1
        self.decoder_reduction_factor = decoder_reduction_factor
        self.speech_pad_id = speech_pad_id
        self.speech_unk_id = speech_unk_id
        self.speech_bos_id = speech_bos_id
        self.speech_eos_id = speech_eos_id
        self.filter_by_source_target_text_ratio = filter_by_source_target_text_ratio
        self.source_target_text_ratio_limit = source_target_text_ratio_limit
        self.sample_rate = sample_rate

        # To be consistent with SALM text processor
        self.text_processor.add_sep = False
        self.text_processor.max_seq_length = (
            4096  # Set this to a large number for since the speech sequence can be long
        )
        self.t5_style = t5_style

    def __getitem__(self, cuts) -> dict[str, torch.Tensor | list[str] | dict]:
        cuts = cuts.sort_by_duration()

        logging.debug(f"Len: {len(cuts)}")

        metadata = []
        instructions, instruction_lengths = [], []
        source_texts, source_text_lengths = [], []  # Not used in the current implementation
        target_texts, target_text_lengths = [], []
        remove_ids = []
        for id, cut in enumerate(cuts):
            metadata.append({'audio_filepath': cut.id + '.wav'})

            instruction = self.text_processor._process_example(context=cut.supervisions[0].text, output="")
            instruction, instruction_length = torch.as_tensor(instruction["input_ids"]), torch.as_tensor(
                len(instruction["input_ids"])
            )

            source_text = self.text_processor._process_example(context=cut.supervisions[1].text, output="")
            source_text, source_text_length = torch.as_tensor(source_text["input_ids"]), torch.as_tensor(
                len(source_text["input_ids"])
            )

            target_text = self.text_processor._process_example(context=cut.supervisions[2].text, output="")
            # -1 to remove the eos token added by the text processor
            target_text, target_text_length = torch.as_tensor(target_text["input_ids"][:-1]), torch.as_tensor(
                len(target_text["input_ids"]) - 1
            )

            if self.filter_by_source_target_text_ratio:
                if (
                    source_text_length / target_text_length > self.source_target_text_ratio_limit
                    or target_text_length / source_text_length > self.source_target_text_ratio_limit
                ):
                    remove_ids.append(id)
                    continue

            instructions.append(instruction)
            instruction_lengths.append(instruction_length)
            source_texts.append(source_text)
            source_text_lengths.append(source_text_length)
            target_texts.append(target_text)
            target_text_lengths.append(target_text_length)

        # cuts = [c for i, c in enumerate(cuts) if i not in remove_ids]

        audio, audio_lens, cuts = self.load_audio(cuts)
        # TODO
        # AudioSamples does not work if the audio files in the CutSet has different sampling rates
        # audio, audio_lens, cuts = zip(*[self.load_audio(CutSet([c])) for c in cuts])
        # Resample audio waveform here since cuts.resample causes core dump sometimes
        # cuts_sample_rates = [c.recording.sampling_rate for c in cuts]
        # import torchaudio
        # audio = [torchaudio.functional.resample(a, orig_sample_rate, self.sample_rate).squeeze(0) for a, orig_sample_rate in zip(audio, cuts_sample_rates)]
        # audio_lens = (torch.IntTensor(audio_lens) * (self.sample_rate / torch.IntTensor(cuts_sample_rates))).int()
        # audio = collate_vectors(audio, max_length=max(audio_lens), padding_value=0.0)
        # cuts = CutSet([c[0] for c in cuts])

        audio_ratio = []
        for id, cut in enumerate(cuts):
            audio_ratio.append(1.0)

        for _, cut in enumerate(cuts):
            if hasattr(cut, self.context_key):
                cut.context = getattr(cut, self.context_key)
            elif hasattr(cut, self.default_context_key):
                cut.context = getattr(cut, self.default_context_key)
            else:
                cut.context = self.default_context

        text_pad_id = self.text_processor.pad_id
        text_unk_id = self.text_processor.unk_id
        text_bos_id = self.text_processor.bos_id
        text_eos_id = self.text_processor.eos_id

        def get_3d_empty_tensor(batch_size, length, text_fill_id, speech_fill_id):
            return torch.cat(
                [
                    torch.full((batch_size, length, 1), text_fill_id),
                    torch.full(
                        (batch_size, length, self.n_speech_codebooks * self.decoder_reduction_factor), speech_fill_id
                    ),
                ],
                axis=2,
            )

        def collate_and_pad(inputs):
            token_lengths = [len(seq) for seq in inputs]
            max_length = max(token_lengths)
            assert len(inputs[0].shape) < 3
            if len(inputs[0].shape) < 2:
                if self.pad_to_max_length:
                    max_length = self.max_seq_length
                else:
                    max_length = min(self.max_seq_length, ceil_to_nearest(max_length, 8))

                tokens = collate_vectors(inputs, max_length=max_length, padding_value=text_pad_id)
            else:
                tokens = get_3d_empty_tensor(len(inputs), max_length, text_pad_id, self.speech_pad_id)
                for i in range(len(tokens)):
                    tokens[i, : token_lengths[i], :] = inputs[i]
            return tokens, torch.LongTensor(token_lengths)

        features_lens = torch.tensor(
            [cut.target_codes.shape[0] // self.decoder_reduction_factor for cut in cuts], dtype=torch.int
        )
        # +1 for the eos tensor
        target_codec = get_3d_empty_tensor(len(cuts), max(features_lens).item() + 1, text_pad_id, self.speech_pad_id)
        eos_tensor = torch.full(
            (1, self.n_speech_codebooks * self.decoder_reduction_factor + 1), self.speech_eos_id
        ).to(torch.int)
        eos_tensor[:, 0] = self.text_processor.unk_id
        # Loop through cuts and build target_codec
        for i, cut in enumerate(cuts):
            feat_i = cut.target_codes.load()
            target_codec[i, : feat_i.shape[0], 0] = text_unk_id
            feat_i = feat_i[: features_lens[i] * self.decoder_reduction_factor, : self.n_speech_codebooks]
            feat_i = feat_i.reshape((-1, self.n_speech_codebooks * self.decoder_reduction_factor))
            target_codec[i, : feat_i.shape[0], 1:] = torch.tensor(feat_i)
            target_codec[i, feat_i.shape[0], :] = eos_tensor

        target_codec = target_codec.to(torch.int)

        source_texts, source_text_lengths = collate_and_pad(source_texts)

        def _convert_text_to_3d_tensor(texts, include_eos=True):
            texts, text_lengths = collate_and_pad(texts)
            texts_expanded = get_3d_empty_tensor(texts.shape[0], texts.shape[1] + 1, text_pad_id, self.speech_pad_id)
            for i, text_length in enumerate(text_lengths):
                texts_expanded[i, :text_length, 0] = texts[i, :text_length]
                texts_expanded[i, :text_length, 1:] = self.speech_unk_id
                eos_tensor = torch.full(
                    (1, self.n_speech_codebooks * self.decoder_reduction_factor + 1), self.speech_eos_id
                ).to(torch.int)
                eos_tensor[:, 0] = self.text_processor.eos_id

                texts_expanded[i, text_length, :] = eos_tensor
            if not include_eos:
                texts_expanded = texts_expanded[:, :-1]
            return texts, text_lengths, texts_expanded

        target_texts, target_text_lengths, target_texts_expanded = _convert_text_to_3d_tensor(target_texts)
        target_texts = target_texts_expanded
        instructions, instruction_lengths, instructions_expanded_no_eos = _convert_text_to_3d_tensor(
            instructions, include_eos=False
        )

        # answers = torch.concat([speaker_context, bos_tensor, target_codec], 1)

        if getattr(cut, "s2s", False):
            # Add 1 for eos token
            token_list = [
                torch.concat([tt[: ttl + 1], tc[: tcl + 1]], 0)
                for tt, ttl, tc, tcl in zip(target_texts, target_text_lengths, target_codec, features_lens)
            ]
            if not self.t5_style:
                token_list = [
                    torch.concat([it[:itl], tt], 0)
                    for tt, it, itl in zip(token_list, instructions_expanded_no_eos, instruction_lengths)
                ]
            tokens, _ = collate_and_pad(token_list)

            # speech_loss_mask = torch.logical_and((tokens[:, :, 1:] != self.speech_unk_id), (tokens[:, :, 1:] != self.speech_pad_id))
            # text_loss_mask = torch.logical_and((tokens[:, :, 0:1] != text_unk_id), (tokens[:, :, 0:1] != text_pad_id))
            speech_loss_mask = tokens[:, :, 1:] != self.speech_pad_id
            text_loss_mask = tokens[:, :, 0:1] != text_pad_id
            if not self.t5_style:
                for itl in instruction_lengths:
                    speech_loss_mask[:, :itl, :] = False
                    text_loss_mask[:, :itl, :] = False
            loss_mask = torch.cat([text_loss_mask, speech_loss_mask], 2)
            full_lengths = target_text_lengths + 1 + features_lens + 1 + instruction_length

        elif getattr(cut, "direct_s2s", False):
            # Add 1 for eos token
            # tt[0] is the bos token
            token_list = [
                torch.concat([tt[:1], tc[: tcl + 1]], 0)
                for tt, tc, tcl in zip(target_texts, target_codec, features_lens)
            ]
            if not self.t5_style:
                token_list = [
                    torch.concat([it[:itl], tt], 0)
                    for tt, it, itl in zip(token_list, instructions_expanded_no_eos, instruction_lengths)
                ]
            tokens, _ = collate_and_pad(token_list)

            speech_loss_mask = tokens[:, :, 1:] != self.speech_pad_id
            text_loss_mask = tokens[:, :, 0:1] != text_pad_id
            if not self.t5_style:
                for itl in instruction_lengths:
                    speech_loss_mask[:, :itl, :] = False
                    text_loss_mask[:, :itl, :] = False
            loss_mask = torch.cat([text_loss_mask, speech_loss_mask], 2)
            full_lengths = 1 + features_lens + 1 + instruction_length
        elif getattr(cut, "s2t", False):
            # Add 1 for eos token
            token_list = [tt[: ttl + 1] for tt, ttl in zip(target_texts, target_text_lengths)]
            if not self.t5_style:
                token_list = [
                    torch.concat([it[:itl], tt], 0)
                    for tt, it, itl in zip(token_list, instructions_expanded_no_eos, instruction_lengths)
                ]
            tokens, _ = collate_and_pad(token_list)

            speech_loss_mask = torch.zeros(tokens.shape[0], tokens.shape[1] - 1, tokens.shape[2])
            text_loss_mask = tokens[:, :, 0:1] != text_pad_id
            if not self.t5_style:
                for itl in instruction_lengths:
                    speech_loss_mask[:, :itl, :] = False
                    text_loss_mask[:, :itl, :] = False
            loss_mask = torch.cat([text_loss_mask, speech_loss_mask], 2)
            full_lengths = target_text_lengths + 1 + instruction_length
        # Start from index 1 since the first token will not be used as a label
        loss_mask = loss_mask[:, 1:, :]

        # Merge batch
        return_batch = {
            "sample_ids": list(cuts.ids),
            "audio_signal": audio,
            "audio_signal_length": audio_lens,
            "audio_ratio": torch.FloatTensor(audio_ratio),
            "metadata": metadata,
            # For forward
            "instructions": instructions,
            "instruction_lengths": instruction_lengths,
            "tokens": tokens[:, :-1, :],
            "tokens_length": full_lengths - 1,
            "labels": tokens[:, 1:, :],
            "loss_mask": loss_mask,
            # For validation mainly
            "source_texts": source_texts,
            "target_texts": target_texts,
            "target_text_lengths": target_text_lengths,
            "answers": tokens[:, 1:, :],
        }

        return return_batch


def collate_text_data(
    cuts,
    default_context: str,
    text_processor: TextProcessing,
    tokens_to_generate: int,
    pad_to_max_length: bool,
    max_seq_length: int,
) -> dict:
    """Perform text collation equivalent to nemo/collections/multimodal/data/audio_text_qa_dataset.py:121"""
    batch_size = len(cuts)
    pad_id = text_processor.pad_id
    examples = [
        {
            k: torch.as_tensor(v)
            for k, v in text_processor._process_example(
                context=cut.context,
                output=cut.supervisions[0].text,
            ).items()
        }
        for cut in cuts
    ]
    fields = as_dict(examples)

    def get_max_len(input_list):
        return max([len(x) for x in input_list])

    max_length = tokens_to_generate + max(
        get_max_len(fields["input_ids"]), get_max_len(fields["context_ids"]), get_max_len(fields["answer_ids"])
    )
    # increase max length to nearest multiple of 4 or 8
    if pad_to_max_length:
        max_length = max_seq_length
    else:
        max_length = min(max_seq_length, ceil_to_nearest(max_length, 8))

    all_tokens = collate_vectors(fields["input_ids"], max_length=max_length, padding_value=pad_id)
    full_lengths = torch.LongTensor([len(item) for item in fields["input_ids"]])

    assert max_length <= max_seq_length, f"{max_length=} <= {max_seq_length=}"

    return {
        "tokens": all_tokens[:, :-1],
        "tokens_length": full_lengths - 1,
        "labels": all_tokens[:, 1:],
        "loss_mask": collate_vectors(
            [torch.as_tensor(build_loss_mask(item)) for item in examples], max_length=max_length, padding_value=0
        )[:, 1:],
        "position_ids": torch.arange(max_length, dtype=torch.long).repeat(batch_size, 1),
        "contexts": collate_vectors(fields["context_ids"], max_length=max_length, padding_value=pad_id),
        "context_lengths": torch.LongTensor([len(seq) for seq in fields["context_ids"]]),
        "answers": collate_vectors(fields["answer_ids"], max_length=max_length, padding_value=pad_id),
        "max_length": torch.LongTensor([max_length] * batch_size),
        "context_ids": fields["context_ids"],
    }


def as_dict(arg: list[dict]) -> dict[str, list]:
    return {k: [item[k] for item in arg] for k in arg[0].keys()}
