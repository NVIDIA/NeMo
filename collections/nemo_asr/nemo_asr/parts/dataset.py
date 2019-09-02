# Taken straight from Patter https://github.com/ryanleary/patter
# TODO: review, and copyright and fix/add comments
import torch
from torch.utils.data import Dataset

from .manifest import Manifest


def seq_collate_fn(batch):
    def find_max_len(seq, index):
        max_len = -1
        for item in seq:
            if item[index].size(0) > max_len:
                max_len = item[index].size(0)
        return max_len

    batch_size = len(batch)

    audio_signal, audio_lengths = None, None
    if batch[0][0] is not None:
        max_audio_len = find_max_len(batch, 0)

        audio_signal = torch.zeros(batch_size, max_audio_len,
                                   dtype=torch.float)
        audio_lengths = []
        for i, s in enumerate(batch):
            audio_signal[i].narrow(0, 0, s[0].size(0)).copy_(s[0])
            audio_lengths.append(s[1])
        audio_lengths = torch.tensor(audio_lengths, dtype=torch.long)

    max_transcript_len = find_max_len(batch, 2)

    transcript = torch.zeros(batch_size, max_transcript_len, dtype=torch.long)
    transcript_lengths = []
    for i, s in enumerate(batch):
        transcript[i].narrow(0, 0, s[2].size(0)).copy_(s[2])
        transcript_lengths.append(s[3])
    transcript_lengths = torch.tensor(transcript_lengths, dtype=torch.long)

    return audio_signal, audio_lengths, transcript, transcript_lengths


def audio_seq_collate_fn(batch):
    """
    collate a batch (iterable of (sample tensor, label tensor) tuples) into
    properly shaped data tensors
    :param batch:
    :return: inputs (batch_size, num_features, seq_length), targets,
    input_lengths, target_sizes
    """
    # sort batch by descending sequence length (for packed sequences later)
    batch.sort(key=lambda x: -x[0].size(0))
    minibatch_size = len(batch)

    # init tensors we need to return
    inputs = torch.zeros(minibatch_size, batch[0][0].size(0))
    input_lengths = torch.zeros(minibatch_size, dtype=torch.long)
    target_sizes = torch.zeros(minibatch_size, dtype=torch.long)
    targets = []
    metadata = []

    # iterate over minibatch to fill in tensors appropriately
    for i, sample in enumerate(batch):
        input_lengths[i] = sample[0].size(0)
        inputs[i].narrow(0, 0, sample[0].size(0)).copy_(sample[0])
        target_sizes[i] = len(sample[1])
        targets.extend(sample[1])
        metadata.append(sample[2])
    targets = torch.tensor(targets, dtype=torch.long)
    return inputs, targets, input_lengths, target_sizes, metadata


class AudioDataset(Dataset):
    def __init__(self, manifest_filepath, labels, featurizer,
                 max_duration=None,
                 min_duration=None, max_utts=0, normalize=True,
                 trim=False, eos_id=None, logger=False, load_audio=True):
        """
        Dataset that loads tensors via a json file containing paths to audio
        files, transcripts, and durations
        (in seconds). Each new line is a different sample. Example below:

        {"audio_filepath": "/path/to/audio.wav", "text_filepath":
        "/path/to/audio.txt", "duration": 23.147}
        ...
        {"audio_filepath": "/path/to/audio.wav", "text": "the
        transcription", offset": 301.75, "duration": 0.82, "utt":
        "utterance_id",
        "ctm_utt": "en_4156", "side": "A"}

        Args:
            manifest_filepath: Path to manifest json as described above. Can
            be coma-separated paths.
            labels: String containing all the possible characters to map to
            featurizer: Initialized featurizer class that converts paths of
            audio to feature tensors
            max_duration: If audio exceeds this length, do not include in
            dataset
            min_duration: If audio is less than this length, do not include
            in dataset
            max_utts: Limit number of utterances
            normalize: whether to normalize transcript text (default): True
            eos_id: Id of end of sequence symbol to append if not None
            load_audio: Boolean flag indicate whether do or not load audio
        """
        m_paths = manifest_filepath.split(',')
        self.manifest = Manifest(m_paths, labels,
                                 max_duration=max_duration,
                                 min_duration=min_duration, max_utts=max_utts,
                                 normalize=normalize)
        self.featurizer = featurizer
        self.trim = trim
        self.eos_id = eos_id
        self.load_audio = load_audio
        if logger:
            logger.info(
                "Dataset loaded with {0:.2f} hours. Filtered {1:.2f} "
                "hours.".format(
                    self.manifest.duration / 3600,
                    self.manifest.filtered_duration / 3600))

    def __getitem__(self, index):
        sample = self.manifest[index]
        if self.load_audio:
            duration = sample['duration'] if 'duration' in sample else 0
            offset = sample['offset'] if 'offset' in sample else 0
            features = self.featurizer.process(sample['audio_filepath'],
                                               offset=offset,
                                               duration=duration,
                                               trim=self.trim)
            f, fl = features, torch.tensor(features.shape[0]).long()
            # f = f / (torch.max(torch.abs(f)) + 1e-5)
        else:
            f, fl = None, None

        t, tl = sample["transcript"], len(sample["transcript"])
        if self.eos_id is not None:
            t = t + [self.eos_id]
            tl += 1

        return \
            f, fl, \
            torch.tensor(t).long(), torch.tensor(tl).long()

    def __len__(self):
        return len(self.manifest)
