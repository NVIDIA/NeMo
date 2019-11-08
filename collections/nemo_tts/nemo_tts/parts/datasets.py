# Copyright (c) 2019 NVIDIA Corporation
import torch
from torch.utils.data import Dataset

from nemo_asr.parts.segment import AudioSegment
from .manifest import AudioManifest


class AudioOnlyDataset(Dataset):
    def __init__(self,
                 manifest_filepath,
                 n_segments=0,
                 max_duration=None,
                 min_duration=None,
                 max_utts=0,
                 trim=False,
                 logger=False):
        """TODO:fix docstring
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
        self.manifest = AudioManifest(m_paths,
                                      max_duration=max_duration,
                                      min_duration=min_duration,
                                      max_utts=max_utts)
        self.trim = trim
        self.n_segments = n_segments
        if logger:
            logger.info(
                f"Dataset loaded with {self.manifest.duration / 3600:.2f} "
                f"hours. Filtered {self.manifest.filtered_duration / 3600:.2f}"
                f" hours.")

    def AudioCollateFunc(self, batch):
        def find_max_len(seq, index):
            max_len = -1
            for item in seq:
                if item[index].size(0) > max_len:
                    max_len = item[index].size(0)
            return max_len

        batch_size = len(batch)

        audio_signal, audio_lengths = None, None
        if batch[0][0] is not None:
            if self.n_segments > 0:
                max_audio_len = self.n_segments
            else:
                max_audio_len = find_max_len(batch, 0)

            audio_signal = torch.zeros(
                batch_size, max_audio_len, dtype=torch.float)
            audio_lengths = []
            for i, s in enumerate(batch):
                audio_signal[i].narrow(0, 0, s[0].size(0)).copy_(s[0])
                audio_lengths.append(s[1])
            audio_lengths = torch.tensor(audio_lengths, dtype=torch.long)

        return audio_signal, audio_lengths

    def __getitem__(self, index):
        sample = self.manifest[index]
        features = AudioSegment.segment_from_file(
            sample['audio_filepath'],
            n_segments=self.n_segments,
            trim=self.trim)
        features = torch.tensor(features.samples, dtype=torch.float)
        f, fl = features, torch.tensor(features.shape[0]).long()

        return f, fl

    def __len__(self):
        return len(self.manifest)
