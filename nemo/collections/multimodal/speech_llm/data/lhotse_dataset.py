import torch.utils.data
import random
import logging
from lhotse import CutSet
from lhotse.dataset import AudioSamples
from lhotse.dataset.collation import collate_vectors as collate_vectors_lhotse
from lhotse.dataset.collation import _read_features

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
        ali_score_key: str = "ali_score",
        default_context_key: str = "default_context",
        vocab_sizes: list[int] = [-1],
        speech_pad_id: int = 1001,
        filter_by_source_target_text_ratio: bool = False,
        source_target_text_ratio_limit: float = 1.0,
    ):
        super().__init__()
        self.text_processor = text_processor
        self.load_audio = AudioSamples(fault_tolerant=True)
        self.tokens_to_generate = tokens_to_generate
        self.pad_to_max_length = pad_to_max_length
        self.max_seq_length = max_seq_length

        self.default_context = default_context
        self.context_key = context_key
        self.ali_score_key = ali_score_key
        self.default_context_key = default_context_key

        if len(vocab_sizes) == 1 and vocab_sizes[0] <= 0:
            vocab_sizes = [self.text_processor.tokenizer.vocab_size]
        self.vocab_sizes = vocab_sizes
        self.n_speech_codebooks = len(self.vocab_sizes) - 1
        self.speech_pad_id = speech_pad_id
        self.filter_by_source_target_text_ratio = filter_by_source_target_text_ratio
        self.source_target_text_ratio_limit = source_target_text_ratio_limit

        # To be consistent with SALM text processor
        self.text_processor.add_sep = False

    def __getitem__(self, cuts) -> dict[str, torch.Tensor | list[str] | dict]:
        cuts = cuts.sort_by_duration()

        # remove_ids = []
        # # In case feature loading fails
        # for i, cut in enumerate(cuts):
        #     try:
        #         cut.load_features()
        #     except:
        #         remove_ids.append(i)
        #         continue
        # cuts = [cut for i, cut in enumerate(cuts) if i not in remove_ids]
        logging.debug(f"Len: {len(cuts)}")

        metadata = []
        instructions, instruction_lengths = [], []
        source_texts, source_text_lengths = [], [] # Not used in the current implementation
        target_texts, target_text_lengths = [], []
        remove_ids = []
        for id, cut in enumerate(cuts):
            metadata.append({'audio_filepath': cut.id + '.wav'})

            instruction = self.text_processor._process_example(
                context=cut.supervisions[0].text, output=""
            )
            instruction, instruction_length = torch.as_tensor(instruction["input_ids"]), torch.as_tensor(len(instruction["input_ids"]))

            source_text = self.text_processor._process_example(
                context=cut.supervisions[1].text, output=""
            )
            source_text, source_text_length = torch.as_tensor(source_text["input_ids"]), torch.as_tensor(len(source_text["input_ids"]))

            target_text = self.text_processor._process_example(
                context=cut.supervisions[2].text, output=""
            )
            target_text, target_text_length = torch.as_tensor(target_text["input_ids"]), torch.as_tensor(len(target_text["input_ids"]))

            if self.filter_by_source_target_text_ratio:
                if source_text_length / target_text_length > self.source_target_text_ratio_limit or \
                    target_text_length / source_text_length > self.source_target_text_ratio_limit:
                    remove_ids.append(id)
                    continue

            instructions.append(instruction)
            instruction_lengths.append(instruction_length)
            source_texts.append(source_text)
            source_text_lengths.append(source_text_length)
            target_texts.append(target_text)
            target_text_lengths.append(target_text_length)
        
        cuts = [c for i, c in enumerate(cuts) if i not in remove_ids]
        cuts = CutSet(cuts)

        audio, audio_lens, cuts = self.load_audio(cuts)

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
            if hasattr(cut, self.ali_score_key):
                cut.ali_score = getattr(cut, self.ali_score_key)

        # Now handle TTS if any
        text_pad_id = self.text_processor.pad_id
        text_unk_id = self.text_processor.unk_id # filled in the text axis when the model output speech codecs

        def get_3d_fully_padded_tensor(batch_size, length):
            return torch.cat(
                [
                    torch.full((batch_size, length, 1), text_pad_id),
                    torch.full((batch_size, length, self.n_speech_codebooks), self.speech_pad_id),
                ], axis=2
            )
        
        def collate_and_pad(inputs):
            token_lengths = [len(seq) for seq in inputs]
            max_length = self.tokens_to_generate + max(token_lengths)
            assert len(inputs[0].shape) < 3
            if len(inputs[0].shape) < 2:
                if self.pad_to_max_length:
                    max_length = self.max_seq_length
                else:
                    max_length = min(self.max_seq_length, ceil_to_nearest(max_length, 8))

                tokens = collate_vectors(inputs, max_length=max_length, padding_value=text_pad_id)
            else:
                tokens = get_3d_fully_padded_tensor(len(inputs), max_length)
                for i in range(len(tokens)):
                    tokens[i, :token_lengths[i], :] = inputs[i]
            return tokens, torch.LongTensor(token_lengths)
            
        features_lens = torch.tensor([cut.target_codes.shape[0] for cut in cuts], dtype=torch.int)
        target_codec = get_3d_fully_padded_tensor(len(cuts), max(features_lens).item()+1)
        eos_tensor = torch.full((1, self.n_speech_codebooks+1), self.speech_pad_id).to(torch.int)
        eos_tensor[:,0] = self.text_processor.eos_id
        # Loop through cuts and build target_codec, label, and context tensors
        speaker_context_list = []
        for i, cut in enumerate(cuts):
            feat_i = cut.target_codes.load()
            target_codec[i,:feat_i.shape[0],0] = text_unk_id
            target_codec[i,:feat_i.shape[0],1:] = torch.tensor(feat_i)[:, :self.n_speech_codebooks]
            target_codec[i, feat_i.shape[0], :] = eos_tensor
            speaker_context = cut.load_context()
            # take random 3s splice from context
            # TODO: fix hardcode
            rng = random.Random()  # Custom random generator (since random uses fixed seeds). Else context remains fixed
            reference_codec_len = 3 * 86
            reference_codec_len = min(reference_codec_len, speaker_context.shape[0])
            si = rng.randint(0, speaker_context.shape[0] - reference_codec_len)
            speaker_context = speaker_context[si : si + reference_codec_len, :self.n_speech_codebooks]
            speaker_context_list.append(torch.tensor(speaker_context))

        target_codec = target_codec.to(torch.int)
        speaker_context = torch.stack(speaker_context_list).to(torch.int) # Not used in the current implementation
        speaker_context = torch.cat([torch.full((speaker_context.shape[0], speaker_context.shape[1], 1), text_unk_id, dtype=speaker_context.dtype), speaker_context], axis=2)

        instructions, instruction_lengths = collate_and_pad(instructions)
        source_texts, source_text_lengths = collate_and_pad(source_texts)
        # Just target text data
        target_texts, target_text_lengths = collate_and_pad(target_texts)
        target_texts_expanded = get_3d_fully_padded_tensor(target_texts.shape[0], target_texts.shape[1])
        target_texts_expanded[:, :, 0] = target_texts
        target_texts = target_texts_expanded

        # bos_tensor = torch.full((len(cuts), 1, self.n_speech_codebooks+1), self.speech_pad_id).to(torch.int)
        # bos_tensor[:,:,0] = self.text_processor.bos_id
        
        # answers = torch.concat([speaker_context, bos_tensor, target_codec], 1)

        if getattr(cut, "s2st", False):
            # Add 1 for eos token
            token_list = [torch.concat([tt[:ttl], tc[:tcl+1]], 0) for tt, ttl, tc, tcl in zip(target_texts, target_text_lengths, target_codec, features_lens)]
            tokens, _ = collate_and_pad(token_list)
            # -1 since the first token will not be used as a label
            loss_mask = torch.zeros(tokens.shape[0], tokens.shape[1]-1, tokens.shape[2])
            for i in range(len(tokens)):
                # loss_mask[i, :target_text_lengths[i]-1, 0] = 1
                # loss_mask[i, target_text_lengths[i]-1:target_text_lengths[i]+features_lens[i], 1:] = 1
                loss_mask[i, :target_text_lengths[i]-1, :] = 1
                loss_mask[i, target_text_lengths[i]-1:target_text_lengths[i]+features_lens[i], :] = 1
        elif getattr(cut, "s2tt", False):
            token_list = [tt[:ttl] for tt, ttl in zip(target_texts, target_text_lengths)]
            tokens, _ = collate_and_pad(token_list)
            loss_mask = torch.zeros(tokens.shape[0], tokens.shape[1]-1, tokens.shape[2])
            for i in range(len(tokens)):
                loss_mask[i, :target_text_lengths[i]-1, 0] = 1

        tokens[:,:,1:] = speech_codec_id_to_token_id(tokens[:,:,1:], self.n_speech_codebooks, self.vocab_sizes)

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

def speech_codec_id_to_token_id(speech_codec, n_speech_codebooks, codebook_sizes):
    """
    Convert raw speech codec ids to tokens ids -- the token table size will be sum(codebook_sizes)

    Args:
        speech_codec: a tensor of shape [batch_size, seq_len, n_speech_codebooks] that contains raw speech codec which takes values from 0 to 1023 if NeMo codec is used
        n_speech_codebooks: the number of speech codebooks
        codebook_sizes: a list of integers of length n_speech_codebooks+1, with codebook_sizes[0] being the size of text vocab and codebook_sizes[1:] being the size of the speech codebooks
    """    
    for i in range(n_speech_codebooks):
        speech_codec[:, :, i] += sum(codebook_sizes[:i+1])
    return speech_codec

def token_id_to_speech_codec_id(speech_tokens, n_speech_codebooks, codebook_sizes):
    """
    Convert tokens ids back to raw speech codec ids

    Args:
        speech_tokens: a tensor of shape [batch_size, seq_len, n_speech_codebooks]
        n_speech_codebooks: the number of speech codebooks
        codebook_sizes: a list of integers of length n_speech_codebooks+1, with codebook_sizes[0] being the size of text vocab and codebook_sizes[1:] being the size of the speech codebooks
    """    
    for i in range(n_speech_codebooks):
        speech_tokens[:, :, i:] -= codebook_sizes[i]
    return speech_tokens