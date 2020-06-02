# Audio dataset and corresponding functions taken from Patter
# https://github.com/ryanleary/patter
# TODO: review, and copyright and fix/add comments
import os

import kaldi_io
import torch
from torch.utils.data import Dataset

from nemo import logging
from nemo.collections.asr.parts import collections, parsers


def seq_collate_fn(batch, token_pad_value=0):
    """collate batch of audio sig, audio len, tokens, tokens len

    Args:
        batch (Optional[FloatTensor], Optional[LongTensor], LongTensor,
               LongTensor):  A tuple of tuples of signal, signal lengths,
               encoded tokens, and encoded tokens length.  This collate func
               assumes the signals are 1d torch tensors (i.e. mono audio).

    """
    _, audio_lengths, _, tokens_lengths = zip(*batch)
    max_audio_len = 0
    has_audio = audio_lengths[0] is not None
    if has_audio:
        max_audio_len = max(audio_lengths).item()
    max_tokens_len = max(tokens_lengths).item()

    audio_signal, tokens = [], []
    for sig, sig_len, tokens_i, tokens_i_len in batch:
        if has_audio:
            sig_len = sig_len.item()
            if sig_len < max_audio_len:
                pad = (0, max_audio_len - sig_len)
                sig = torch.nn.functional.pad(sig, pad)
            audio_signal.append(sig)
        tokens_i_len = tokens_i_len.item()
        if tokens_i_len < max_tokens_len:
            pad = (0, max_tokens_len - tokens_i_len)
            tokens_i = torch.nn.functional.pad(tokens_i, pad, value=token_pad_value)
        tokens.append(tokens_i)

    if has_audio:
        audio_signal = torch.stack(audio_signal)
        audio_lengths = torch.stack(audio_lengths)
    else:
        audio_signal, audio_lengths = None, None
    tokens = torch.stack(tokens)
    tokens_lengths = torch.stack(tokens_lengths)

    return audio_signal, audio_lengths, tokens, tokens_lengths


def fixed_seq_collate_fn(batch, fixed_length=16000):
    """collate batch of audio sig, audio len, tokens, tokens len

    Args:
        batch (Optional[FloatTensor], Optional[LongTensor], LongTensor,
               LongTensor):  A tuple of tuples of signal, signal lengths,
               encoded tokens, and encoded tokens length.  This collate func
               assumes the signals are 1d torch tensors (i.e. mono audio).
        fixed_length (Optional[int]): length of input signal to be considered

    """
    _, audio_lengths, _, tokens_lengths = zip(*batch)

    has_audio = audio_lengths[0] is not None
    fixed_length = min(fixed_length, max(audio_lengths))

    audio_signal, tokens = [], []
    for sig, sig_len, tokens_i, _ in batch:
        if has_audio:
            sig_len = sig_len.item()
            chunck_len = sig_len - fixed_length
            if chunck_len < 0:
                # pad = (0,fixed_length-sig_len)
                # signal = torch.nn.functional.pad(sig,pad)
                repeat = fixed_length // sig_len
                rem = fixed_length % sig_len
                sub = sig[-rem:] if rem > 0 else torch.tensor([])
                rep_sig = torch.cat(repeat * [sig])
                signal = torch.cat((rep_sig, sub))
                # print(sig_len,repeat,rem,len(sub),len(rep_sig),len(signal))
            else:
                start_idx = torch.randint(0, chunck_len, (1,)) if chunck_len else torch.tensor(0)
                end_idx = start_idx + fixed_length
                signal = sig[start_idx:end_idx]

            audio_signal.append(signal)
        tokens.append(tokens_i)

    if has_audio:
        audio_signal = torch.stack(audio_signal)
        audio_lengths = torch.stack(audio_lengths)
    else:
        audio_signal, audio_lengths = None, None
    tokens = torch.stack(tokens)
    tokens_lengths = torch.stack(tokens_lengths)

    return audio_signal, audio_lengths, tokens, tokens_lengths


def audio_seq_collate_fn(batch):
    """
    Collate a batch (iterable of (sample tensor, label tensor) tuples) into
    properly shaped data tensors
    :param batch:
    :return: inputs (batch_size, num_features, seq_length), targets,
    input_lengths, target_sizes
    """
    # sort batch by descending sequence length (for packed sequences later)
    batch.sort(key=lambda x: x[0].size(0), reverse=True)

    # init tensors we need to return
    inputs = []
    input_lengths = []
    target_sizes = []
    targets = []
    metadata = []

    # iterate over minibatch to fill in tensors appropriately
    for i, sample in enumerate(batch):
        input_lengths.append(sample[0].size(0))
        inputs.append(sample[0])
        target_sizes.append(len(sample[1]))
        targets.extend(sample[1])
        metadata.append(sample[2])
    targets = torch.tensor(targets, dtype=torch.long)
    inputs = torch.stack(inputs)
    input_lengths = torch.stack(input_lengths)
    target_sizes = torch.stack(target_sizes)

    return inputs, targets, input_lengths, target_sizes, metadata


class AudioDataset(Dataset):
    """
    Dataset that loads tensors via a json file containing paths to audio
    files, transcripts, and durations (in seconds). Each new line is a
    different sample. Example below:

    {"audio_filepath": "/path/to/audio.wav", "text_filepath":
    "/path/to/audio.txt", "duration": 23.147}
    ...
    {"audio_filepath": "/path/to/audio.wav", "text": "the
    transcription", "offset": 301.75, "duration": 0.82, "utt":
    "utterance_id", "ctm_utt": "en_4156", "side": "A"}

    Args:
        manifest_filepath: Path to manifest json as described above. Can
            be comma-separated paths.
        labels: String containing all the possible characters to map to
        featurizer: Initialized featurizer class that converts paths of
            audio to feature tensors
        max_duration: If audio exceeds this length, do not include in dataset
        min_duration: If audio is less than this length, do not include
            in dataset
        max_utts: Limit number of utterances
        blank_index: blank character index, default = -1
        unk_index: unk_character index, default = -1
        normalize: whether to normalize transcript text (default): True
        bos_id: Id of beginning of sequence symbol to append if not None
        eos_id: Id of end of sequence symbol to append if not None
        load_audio: Boolean flag indicate whether do or not load audio
        add_misc: True if add adiditional info dict.
    """

    def __init__(
        self,
        manifest_filepath,
        labels,
        featurizer,
        max_duration=None,
        min_duration=None,
        max_utts=0,
        blank_index=-1,
        unk_index=-1,
        normalize=True,
        trim=False,
        bos_id=None,
        eos_id=None,
        load_audio=True,
        parser='en',
        add_misc=False,
    ):
        self.collection = collections.ASRAudioText(
            manifests_files=manifest_filepath.split(','),
            parser=parsers.make_parser(
                labels=labels, name=parser, unk_id=unk_index, blank_id=blank_index, do_normalize=normalize,
            ),
            min_duration=min_duration,
            max_duration=max_duration,
            max_number=max_utts,
        )

        self.featurizer = featurizer
        self.trim = trim
        self.eos_id = eos_id
        self.bos_id = bos_id
        self.load_audio = load_audio
        self._add_misc = add_misc

    def __getitem__(self, index):
        sample = self.collection[index]
        if self.load_audio:
            offset = sample.offset

            if offset is None:
                offset = 0

            features = self.featurizer.process(
                sample.audio_file, offset=offset, duration=sample.duration, trim=self.trim,
            )
            f, fl = features, torch.tensor(features.shape[0]).long()
        else:
            f, fl = None, None

        t, tl = sample.text_tokens, len(sample.text_tokens)
        if self.bos_id is not None:
            t = [self.bos_id] + t
            tl += 1
        if self.eos_id is not None:
            t = t + [self.eos_id]
            tl += 1

        output = f, fl, torch.tensor(t).long(), torch.tensor(tl).long()

        if self._add_misc:
            misc = dict()
            misc['id'] = sample.id
            misc['text_raw'] = sample.text_raw
            misc['speaker'] = sample.speaker
            output = (output, misc)

        return output

    def __len__(self):
        return len(self.collection)


class KaldiFeatureDataset(Dataset):
    """
    Dataset that provides basic Kaldi-compatible dataset loading. Assumes that
    the files `feats.scp`, `text`, and (optionally) `utt2dur` exist, as well
    as the .ark files that `feats.scp` points to.

    Args:
        kaldi_dir: Path to directory containing the aforementioned files.
        labels: All possible characters to map to.
        min_duration: If audio is shorter than this length, drop it. Only
            available if the `utt2dur` file exists.
        max_duration: If audio is longer than this length, drop it. Only
            available if the `utt2dur` file exists.
        max_utts: Limits the number of utterances.
        unk_index: unk_character index, default = -1
        blank_index: blank character index, default = -1
        normalize: whether to normalize transcript text. Defaults to True.
        eos_id: Id of end of sequence symbol to append if not None.
    """

    def __init__(
        self,
        kaldi_dir,
        labels,
        min_duration=None,
        max_duration=None,
        max_utts=0,
        unk_index=-1,
        blank_index=-1,
        normalize=True,
        eos_id=None,
    ):
        self.eos_id = eos_id
        self.unk_index = unk_index
        self.blank_index = blank_index
        self.labels_map = {label: i for i, label in enumerate(labels)}

        data = []
        duration = 0.0
        filtered_duration = 0.0

        # Read Kaldi features (MFCC, PLP) using feats.scp
        feats_path = os.path.join(kaldi_dir, 'feats.scp')
        id2feats = {utt_id: torch.from_numpy(feats) for utt_id, feats in kaldi_io.read_mat_scp(feats_path)}

        # Get durations, if utt2dur exists
        utt2dur_path = os.path.join(kaldi_dir, 'utt2dur')
        id2dur = {}
        if os.path.exists(utt2dur_path):
            with open(utt2dur_path, 'r') as f:
                for line in f:
                    utt_id, dur = line.split()
                    id2dur[utt_id] = float(dur)
        elif max_duration or min_duration:
            raise ValueError(
                f"KaldiFeatureDataset max_duration or min_duration is set but"
                f" utt2dur file not found in {kaldi_dir}."
            )
        else:
            logging.info(
                f"Did not find utt2dur when loading data from " f"{kaldi_dir}. Skipping dataset duration calculations."
            )

        # Match transcripts to features
        text_path = os.path.join(kaldi_dir, 'text')
        parser = parsers.make_parser(labels, 'en', unk_id=unk_index, blank_id=self.blank_index)
        with open(text_path, 'r') as f:
            for line in f:
                split_idx = line.find(' ')
                utt_id = line[:split_idx]

                audio_features = id2feats.get(utt_id)

                if audio_features is not None:

                    text = line[split_idx:].strip()
                    if normalize:
                        # TODO: WTF?
                        text = parser._normalize(text)

                    dur = id2dur[utt_id] if id2dur else None

                    # Filter by duration if specified & utt2dur exists
                    if min_duration and dur < min_duration:
                        filtered_duration += dur
                        continue
                    if max_duration and dur > max_duration:
                        filtered_duration += dur
                        continue

                    sample = {
                        'utt_id': utt_id,
                        'text': text,
                        'tokens': parser(text),
                        'audio': audio_features.t(),
                        'duration': dur,
                    }

                    data.append(sample)
                    duration += dur

                    if max_utts > 0 and len(data) >= max_utts:
                        logging.warning(f"Stop parsing due to max_utts ({max_utts})")
                        break

        if id2dur:
            # utt2dur durations are in seconds
            logging.info(
                f"Dataset loaded with {duration / 3600 : .2f} hours. "
                f"Filtered {filtered_duration / 3600 : .2f} hours."
            )

        self.data = data

    def __getitem__(self, index):
        sample = self.data[index]
        f = sample['audio']
        fl = torch.tensor(f.shape[1]).long()
        t, tl = sample['tokens'], len(sample['tokens'])

        if self.eos_id is not None:
            t.append(self.eos_id)
            tl += 1

        return f, fl, torch.tensor(t).long(), torch.tensor(tl).long()

    def __len__(self):
        return len(self.data)


class TranscriptDataset(Dataset):
    """A dataset class that reads and returns the text of a file.

    Args:
        path: (str) Path to file with newline separate strings of text
        labels (list): List of string labels to use when to str2int translation
        eos_id (int): Label position of end of string symbol
    """

    def __init__(self, path, labels, bos_id=None, eos_id=None, lowercase=True):
        parser = parsers.make_parser(labels, do_lowercase=lowercase)
        self.texts = collections.FromFileText(path, parser=parser)

        self.bos_id = bos_id
        self.eos_id = eos_id

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        tokenized_text = self.texts[item].tokens
        if self.bos_id:
            tokenized_text = [self.bos_id] + tokenized_text
        if self.eos_id:
            tokenized_text = tokenized_text + [self.eos_id]
        return (
            torch.tensor(tokenized_text, dtype=torch.long),
            torch.tensor(len(tokenized_text), dtype=torch.long),
        )


class AudioLabelDataset(Dataset):
    """
    Dataset that loads tensors via a json file containing paths to audio
    files, command class, and durations (in seconds). Each new line is a
    different sample. Example below:

    {"audio_filepath": "/path/to/audio.wav", "label":
    "label", "duration": 23.147}
    ...
    {"audio_filepath": "/path/to/audio.wav", "label": "label",
    "offset": 301.75, "duration": 0.82}

    Args:
        manifest_filepath: Path to manifest json as described above. Can
            be comma-separated paths.
        labels (Optional[list]): String containing all the possible labels to map to
            if None then automatically picks from ASRSpeechLabel collection.
        featurizer: Initialized featurizer class that converts paths of
            audio to feature tensors
        max_duration: If audio exceeds this length, do not include in dataset
        min_duration: If audio is less than this length, do not include
            in dataset
        trim: Boolean flag whether to trim the audio
        load_audio: Boolean flag indicate whether do or not load audio
    """

    def __init__(
        self,
        manifest_filepath,
        featurizer,
        labels=None,
        max_duration=None,
        min_duration=None,
        trim=False,
        load_audio=True,
    ):
        self.collection = collections.ASRSpeechLabel(
            manifests_files=manifest_filepath.split(','), min_duration=min_duration, max_duration=max_duration,
        )

        self.featurizer = featurizer
        self.trim = trim
        self.load_audio = load_audio

        self.labels = labels if labels else self.collection.uniq_labels
        self.num_commands = len(self.labels)

        self.label2id, self.id2label = {}, {}
        for label_id, label in enumerate(self.labels):
            self.label2id[label] = label_id
            self.id2label[label_id] = label

        for idx in range(len(self.labels[:5])):
            logging.debug(" label id {} and its mapped label {}".format(idx, self.id2label[idx]))

    def __getitem__(self, index):
        sample = self.collection[index]
        if self.load_audio:
            offset = sample.offset

            if offset is None:
                offset = 0

            features = self.featurizer.process(
                sample.audio_file, offset=offset, duration=sample.duration, trim=self.trim
            )
            f, fl = features, torch.tensor(features.shape[0]).long()
        else:
            f, fl = None, None

        t = self.label2id[sample.label]
        tl = 1  # For compatibility with collate_fn used later

        return f, fl, torch.tensor(t).long(), torch.tensor(tl).long()

    def __len__(self):
        return len(self.collection)
