from typing import Callable, Dict, List, Optional, Union

import torch

from nemo.collections.asr.data.audio_to_text import ASRManifestProcessor
from nemo.collections.asr.data.feature_to_label import _audio_feature_collate_fn
from nemo.collections.asr.parts.preprocessing.feature_loader import ExternalFeatureLoader
from nemo.collections.asr.parts.utils.audio_utils import ChannelSelectorType
from nemo.collections.asr.parts.utils.vad_utils import load_speech_segments_from_rttm
from nemo.collections.common import tokenizers
from nemo.collections.common.parts.preprocessing import collections, parsers
from nemo.core.classes import Dataset
from nemo.core.neural_types import *
from nemo.utils import logging


class ASRFeatureManifestProcessor(ASRManifestProcessor):
    def __init__(
        self,
        manifest_filepath: str,
        parser: Union[str, Callable],
        max_duration: Optional[float] = None,
        min_duration: Optional[float] = None,
        max_utts: int = 0,
        bos_id: Optional[int] = None,
        eos_id: Optional[int] = None,
        pad_id: int = 0,
        index_by_file_id: bool = False,
    ):
        super().__init__(
            manifest_filepath, parser, max_duration, min_duration, max_utts, bos_id, eos_id, pad_id, index_by_file_id
        )

        self.collection = collections.ASRFeatureText(
            manifests_files=manifest_filepath,
            parser=parser,
            min_duration=min_duration,
            max_duration=max_duration,
            max_number=max_utts,
            index_by_file_id=index_by_file_id,
        )


class _FeatureTextDataset(Dataset):
    """
    Dataset that loads tensors via a json file containing paths to audio files, transcripts, and durations (in seconds).
    Each new line is a different sample. Example below:
    {"feature_filepath": "/path/to/audio.pt", "text_filepath": "/path/to/audio.txt", 
    "rttm_filepath": "/path/to/audio_rttm.rttm", "duration": 23.147}
    ...
    {"feature_filepath": "/path/to/audio.pt", "text": "the transcription", "offset": 301.75, "duration": 0.82, "utt":
    "utterance_id", "ctm_utt": "en_4156", "side": "A"}
    Args:
        manifest_filepath: Path to manifest json as described above. Can be comma-separated paths.
        parser: Str for a language specific preprocessor or a callable.
        sample_rate (int): Sample rate to resample loaded audio to
        int_values (bool): If true, load samples as 32-bit integers. Defauts to False.
        augmentor (nemo.collections.asr.parts.perturb.AudioAugmentor): An AudioAugmentor object used to augment loaded
            audio
        max_duration: If audio exceeds this length, do not include in dataset
        min_duration: If audio is less than this length, do not include in dataset
        max_utts: Limit number of utterances
        trim: whether or not to trim silence. Defaults to False
        bos_id: Id of beginning of sequence symbol to append if not None
        eos_id: Id of end of sequence symbol to append if not None
        pad_id: Id of pad symbol. Defaults to 0
        return_sample_id (bool): whether to return the sample_id as a part of each sample
        channel_selector (int | Iterable[int] | str): select a single channel or a subset of channels from multi-channel audio. If set to `'average'`, it performs averaging across channels. Disabled if set to `None`. Defaults to `None`. Uses zero-based indexing.
    """

    ZERO_LEVEL_SPEC_DB_VAL = -16.635  # Log-Melspectrogram value for zero signal
    NORM_MODES = ["pre_norm", "post_norm"]
    MASK_MODES = ["zero", "avg", "min"]

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports.
               """
        return {
            'features': NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),
            'feature_length': NeuralType(tuple('B'), LengthsType()),
            'transcripts': NeuralType(('B', 'T'), LabelsType()),
            'transcript_length': NeuralType(tuple('B'), LengthsType()),
            'sample_id': NeuralType(tuple('B'), LengthsType(), optional=True),
        }

    def __init__(
        self,
        manifest_filepath: str,
        parser: Union[str, Callable],
        normalize: str = "post_norm",
        use_rttm: bool = False,
        mask_mode: str = "zero",
        frame_unit_time_secs: float = 0.01,
        sample_rate: Optional[int] = 16000,
        augmentor: 'nemo.collections.asr.parts.perturb.FeatureAugmentor' = None,
        max_duration: Optional[int] = None,
        min_duration: Optional[int] = None,
        max_utts: int = 0,
        trim: bool = False,
        bos_id: Optional[int] = None,
        eos_id: Optional[int] = None,
        pad_id: int = 0,
        return_sample_id: bool = False,
        channel_selector: Optional[ChannelSelectorType] = None,
    ):
        if type(manifest_filepath) == str:
            manifest_filepath = manifest_filepath.split(",")

        self.sample_rate = sample_rate
        self.normalize = normalize
        self.use_rttm = use_rttm
        self.mask_mode = mask_mode

        if normalize is not None and normalize not in self.NORM_MODES:
            raise ValueError(f"`normalize` must be one of {self.NORM_MODES}, got `{normalize}` instead")

        if use_rttm and mask_mode not in self.MASK_MODES:
            raise ValueError(f"`mask_mode` must be one of {self.MASK_MODES}, got `{mask_mode}` instead")

        self.frame_unit_time_secs = frame_unit_time_secs

        self.manifest_processor = ASRFeatureManifestProcessor(
            manifest_filepath=manifest_filepath,
            parser=parser,
            max_duration=max_duration,
            min_duration=min_duration,
            max_utts=max_utts,
            bos_id=bos_id,
            eos_id=eos_id,
            pad_id=pad_id,
        )
        self.featurizer = ExternalFeatureLoader(augmentor=augmentor)
        self.trim = trim
        self.return_sample_id = return_sample_id
        self.channel_selector = channel_selector

    def get_manifest_sample(self, sample_id):
        return self.manifest_processor.collection[sample_id]

    def __getitem__(self, index):
        sample = self.manifest_processor.collection[index]
        offset = sample.offset

        if offset is None:
            offset = 0

        features = self.featurizer.process(sample.feature_file)

        f, fl = features, torch.tensor(features.shape[1]).long()

        t, tl = self.manifest_processor.process_text_by_sample(sample=sample)

        # Feature normalization
        if self.normalize is None:
            if self.use_rttm and sample.rttm_file:
                f = self.mask_features_from_rttm(f, offset, sample.rttm_file, self.ZERO_LEVEL_SPEC_DB_VAL)
        elif self.normalize == "post_norm":
            # (Optional) Masking based on RTTM file
            if self.use_rttm and sample.rttm_file:
                f = self.mask_features_from_rttm(f, offset, sample.rttm_file, self.ZERO_LEVEL_SPEC_DB_VAL)
            f = self.featurizer.normalize_per_feature(f)
        else:
            f = self.featurizer.normalize_per_feature(f)
            if self.use_rttm and sample.rttm_file:
                f = self.mask_features_from_rttm(f, offset, sample.rttm_file, 0.0)  # similar to SpecAug

        if self.return_sample_id:
            output = f, fl, torch.tensor(t).long(), torch.tensor(tl).long(), index
        else:
            output = f, fl, torch.tensor(t).long(), torch.tensor(tl).long()

        return output

    def mask_features_from_rttm(self, features, offset, rttm_file, mask_val):
        segments = load_speech_segments_from_rttm(rttm_file)
        sid = 0
        for i in range(features.size(1)):
            t = offset + i * self.frame_unit_time_secs
            while sid < len(segments) - 1 and segments[sid][1] < t:
                sid += 1
            if t < segments[sid][0] or t > segments[sid][1]:
                features[:, i] = mask_val
        return features

    def __len__(self):
        return len(self.manifest_processor.collection)

    def _collate_fn(self, batch):
        return _audio_feature_collate_fn(
            batch, feat_pad_val=self.ZERO_LEVEL_SPEC_DB_VAL, label_pad_id=self.manifest_processor.pad_id
        )


class FeatureToCharDataset(_FeatureTextDataset):
    def __init__(
        self,
        manifest_filepath: str,
        labels: Union[str, List[str]],
        normalize: str = "post_norm",
        use_rttm: bool = False,
        frame_unit_time_secs: float = 0.01,
        sample_rate: Optional[int] = 16000,
        augmentor: 'nemo.collections.asr.parts.perturb.FeatureAugmentor' = None,
        max_duration: Optional[int] = None,
        min_duration: Optional[int] = None,
        max_utts: int = 0,
        blank_index: int = -1,
        unk_index: int = -1,
        trim: bool = False,
        bos_id: Optional[int] = None,
        eos_id: Optional[int] = None,
        pad_id: int = 0,
        parser: Union[str, Callable] = 'en',
        return_sample_id: bool = False,
        channel_selector: Optional[ChannelSelectorType] = None,
    ):
        self.labels = labels

        parser = parsers.make_parser(
            labels=labels, name=parser, unk_id=unk_index, blank_id=blank_index, do_normalize=normalize
        )

        super().__init__(
            manifest_filepath=manifest_filepath,
            parser=parser,
            normalize=normalize,
            use_rttm=use_rttm,
            frame_unit_time_secs=frame_unit_time_secs,
            sample_rate=sample_rate,
            augmentor=augmentor,
            max_duration=max_duration,
            min_duration=min_duration,
            max_utts=max_utts,
            trim=trim,
            bos_id=bos_id,
            eos_id=eos_id,
            pad_id=pad_id,
            return_sample_id=return_sample_id,
            channel_selector=channel_selector,
        )


class FeatureToBPEDataset(_FeatureTextDataset):
    def __init__(
        self,
        manifest_filepath: str,
        tokenizer: 'nemo.collections.common.tokenizers.TokenizerSpec',
        normalize: str = "post_norm",
        use_rttm: bool = False,
        frame_unit_time_secs: float = 0.01,
        sample_rate: Optional[int] = 16000,
        augmentor: 'nemo.collections.asr.parts.perturb.FeatureAugmentor' = None,
        max_duration: Optional[int] = None,
        min_duration: Optional[int] = None,
        max_utts: int = 0,
        use_start_end_token: bool = True,
        trim: bool = False,
        return_sample_id: bool = False,
        channel_selector: Optional[ChannelSelectorType] = None,
    ):
        if use_start_end_token and hasattr(tokenizer, "bos_id") and tokenizer.bos_id > 0:
            bos_id = tokenizer.bos_id
        else:
            bos_id = None

        if use_start_end_token and hasattr(tokenizer, "eos_id") and tokenizer.eos_id > 0:
            eos_id = tokenizer.eos_id
        else:
            eos_id = None

        if hasattr(tokenizer, "pad_id") and tokenizer.pad_id > 0:
            pad_id = tokenizer.pad_id
        else:
            pad_id = 0

        class TokenizerWrapper:
            def __init__(self, tokenizer):
                if isinstance(tokenizer, tokenizers.aggregate_tokenizer.AggregateTokenizer):
                    self.is_aggregate = True
                else:
                    self.is_aggregate = False
                self._tokenizer = tokenizer

            def __call__(self, *args):
                if isinstance(args[0], List) and self.is_aggregate:
                    t = []
                    for span in args[0]:
                        t.extend(self._tokenizer.text_to_ids(span['str'], span['lang']))
                    return t

                t = self._tokenizer.text_to_ids(*args)
                return t

        super().__init__(
            manifest_filepath=manifest_filepath,
            parser=TokenizerWrapper(tokenizer),
            normalize=normalize,
            use_rttm=use_rttm,
            frame_unit_time_secs=frame_unit_time_secs,
            sample_rate=sample_rate,
            augmentor=augmentor,
            max_duration=max_duration,
            min_duration=min_duration,
            max_utts=max_utts,
            trim=trim,
            bos_id=bos_id,
            eos_id=eos_id,
            pad_id=pad_id,
            return_sample_id=return_sample_id,
            channel_selector=channel_selector,
        )
