# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import io
import numpy as np
import torch
import torchvision
from PIL import Image
import webdataset as wds

from nemo.collections.asr.data.audio_to_text import VALID_FILE_FORMATS as VALID_AUDIO_FILE_FORMATS
from nemo.collections.asr.data.audio_to_text import expand_sharded_filepaths

from nemo.collections.asr.parts.preprocessing.features import WaveformFeaturizer

from nemo.core.classes import IterableDataset
from nemo.utils.distributed import webdataset_split_by_workers
from nemo.collections.common.parts.preprocessing import manifest

__all__ = [
    'get_tarred_audio_visual_text_dataset_from_config',
]

VALID_IMAGE_FILE_FORMATS_SET = {ex for ex, f in Image.registered_extensions().items() if f in Image.OPEN}
VALID_VIDEO_FILE_FORMATS_SET = {'mp4'}



class AVLMAudioVisualTextEntity(object):
    """Class for AVLM dataloader instance."""

    def __init__(self, sid, audio_file, visual_filepath: List[str], duration, context, answer, offset, speaker, orig_sr, lang) -> None:
        """Initialize the AudioTextEntity for a AVLM dataloader instance."""
        self.id = sid
        self.audio_file = audio_file
        """ it is either a single image/video file or a sequence of index named images or empty List"""
        self.visual_files = visual_files
        self.duration = duration
        self.context = context
        self.answer = answer
        self.offset = offset
        self.speaker = speaker
        self.orig_sr = orig_sr
        self.lang = lang

class AVLMAudioVisualText(object):
    """List of audio-transcript text correspondence with preprocessing.

    All of the audio, duration, context, answer are optional.
    If answer is not present, text is treated as the answer.
    """

    def __init__(
        self,
        ids: List[int],
        audio_files: List[str],
        visual_filepaths: List[str],
        durations: List[float],
        context_list: List[str],
        answers: List[str],
        offsets: List[str],
        speakers: List[Optional[int]],
        orig_sampling_rates: List[Optional[int]],
        langs: List[Optional[str]],
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
        max_number: Optional[int] = None,
        do_sort_by_duration: bool = False,
        index_by_file_id: bool = False,
        max_num_samples: Optional[int] = None,
    ):
        """Instantiates audio-context-answer manifest with filters and preprocessing.

        Args:
            ids: List of examples positions.
            audio_files: List of audio files.
            visual_filepaths: List of image/video path(s).
                Each path is either a single file name or an absolute path
            durations: List of float durations.
            context_list: List of raw text transcripts.
            answers: List of raw text transcripts.
            offsets: List of duration offsets or None.
            speakers: List of optional speakers ids.
            orig_sampling_rates: List of original sampling rates of audio files.
            langs: List of language ids, one for eadh sample, or None.
            min_duration: Minimum duration to keep entry with (default: None).
            max_duration: Maximum duration to keep entry with (default: None).
            max_number: Maximum number of samples to collect.
            do_sort_by_duration: True if sort samples list by duration. Not compatible with index_by_file_id.
            index_by_file_id: If True, saves a mapping from filename base (ID) to index in data.
        """

        data, duration_filtered, num_filtered, total_duration = [], 0.0, 0, 0.0
        if index_by_file_id:
            self.mapping = {}

        for id_, audio_file, visual_filepath, duration, offset, context, answer, speaker, orig_sr, lang in zip(
            ids, audio_files, visual_filepaths, durations, offsets, context_list, answers, speakers, orig_sampling_rates, langs
        ):
            # Duration filters.
            if duration is not None:
                curr_min_dur = min(duration) if isinstance(duration, list) else duration
                curr_max_dur = max(duration) if isinstance(duration, list) else duration
                curr_sum_dur = sum(duration) if isinstance(duration, list) else duration
                if min_duration is not None and curr_min_dur < min_duration:
                    duration_filtered += curr_sum_dur
                    num_filtered += 1
                    continue

                if max_duration is not None and curr_max_dur > max_duration:
                    duration_filtered += curr_sum_dur
                    num_filtered += 1
                    continue
                total_duration += curr_sum_dur

            if answer is None:
                duration_filtered += curr_sum_dur
                num_filtered += 1
                continue

            visual_files = []
            if visual_filepath is not None:
                if os.path.isfile(visual_filepath):
                    visual_files = [visual_filepath]
                elif os.path.isdir(visual_filepath):
                    visual_files = [f for f in os.listdir(visual_filepath) if \
                        os.path.splitext(f)[1] in VALID_IMAGE_FILE_FORMATS_SET and \
                        os.path.splitext(os.path.basename(f))[0].isdigit()]
                    # sort the files according to the images' names. Assume names are indexes
                    visual_files.sort(key=lambda x: int(x))

            data.append(
                AVLMAudioVisualTextEntity(id_, audio_file, visual_files, duration, context, answer, offset, speaker, orig_sr, lang)
            )
            if index_by_file_id and (audio_file is not None or visual_files is not []):
                if audio_file is not None:
                    file_id, _ = os.path.splitext(os.path.basename(audio_file))
                else:
                    file_id, _ = os.path.splitext(os.path.basename(visual_files[0]))
                if file_id not in self.mapping:
                    self.mapping[file_id] = []
                self.mapping[file_id].append(len(data) - 1)

            # Max number of entities filter.
            if len(data) == max_number:
                break

        if max_num_samples is not None and not index_by_file_id:
            if max_num_samples <= len(data):
                logging.info(f"Subsampling dataset from {len(data)} to {max_num_samples} samples")
                data = data[:max_num_samples]
            else:
                logging.info(f"Oversampling dataset from {len(data)} to {max_num_samples} samples")
                data = data * (max_num_samples // len(data))
                res_num = max_num_samples % len(data)
                res_data = [data[idx] for idx in np.random.choice(len(data), res_num, replace=False)]
                data.extend(res_data)
        elif max_num_samples is not None and index_by_file_id:
            logging.warning("Tried to subsample dataset by max_num_samples, but cannot since index_by_file_id is set.")

        if do_sort_by_duration:
            if index_by_file_id:
                logging.warning("Tried to sort dataset by duration, but cannot since index_by_file_id is set.")
            else:
                data.sort(key=lambda entity: entity.duration)

        logging.info("Dataset loaded with %d files totalling %.2f hours", len(data), total_duration / 3600)
        logging.info("%d files were filtered totalling %.2f hours", num_filtered, duration_filtered / 3600)

        self.data = data

    def __getitem__(self, idx):
        if idx < 0 or idx > len(self.data):
            raise ValueError(f"index out of range [0,{len(self.data)}), got {idx} instead")
        return self.data[idx]

    def __len__(self):
        return len(self.data)

class AVLMAudioVisualTextCollection(AVLMAudioVisualText):
    """`AVLMAudioVisualText` collector from SpeechLLM json files.

    This collector also keeps backward compatibility with AVLMAudioVisualText.
    """

    def __init__(
        self,
        manifests_files: Union[str, List[str]],
        context_file: Optional[Union[List[str], str]] = None,
        context_key: str = "context",
        answer_key: str = "answer",
        *args,
        **kwargs,
    ):
        """Parse lists of audio files, image/video files, durations and transcripts texts.

        Args:
            manifests_files: Either single string file or list of such -
                manifests to yield items from.
            *args: Args to pass to `AudioText` constructor.
            **kwargs: Kwargs to pass to `AudioText` constructor.
        """
        self.context_key = context_key
        self.answer_key = answer_key

        (
            ids,
            audio_files,
            visual_filepaths,
            durations,
            context_list,
            answers,
            offsets,
        ) = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        speakers, orig_srs, langs = (
            [],
            [],
            [],
        )
        if context_file is not None:
            question_file_list = context_file.split(",") if isinstance(context_file, str) else context_file
            self.context_list = []
            for filepath in question_file_list:
                with open(filepath, 'r') as f:
                    for line in f.readlines():
                        line = line.strip()
                        if line:
                            self.context_list.append(line)
            logging.info(f"Use random text context from {context_file} for {manifests_files}")
        else:
            self.context_list = None

        for item in manifest.item_iter(manifests_files, parse_func=self.__parse_item):
            ids.append(item['id'])
            audio_files.append(item['audio_file'])
            visual_filepaths.append(item['visual_filepath'])
            durations.append(item['duration'])
            context_list.append(item['context'])
            answers.append(item['answer'])
            offsets.append(item['offset'])
            speakers.append(item['speaker'])
            orig_srs.append(item['orig_sr'])
            langs.append(item['lang'])
        super().__init__(
            ids, audio_files, visual_filepaths, durations, context_list, answers, offsets, speakers, orig_srs, langs, *args, **kwargs
        )

    def __parse_item(self, line: str, manifest_file: str) -> Dict[str, Any]:
        item = json.loads(line)

        # Audio file
        if 'audio_filename' in item:
            item['audio_file'] = item.pop('audio_filename')
        elif 'audio_filepath' in item:
            item['audio_file'] = item.pop('audio_filepath')
        elif 'audio_file' not in item:
            item['audio_file'] = None

        # If the audio path is a relative path and does not exist,
        # try to attach the parent directory of manifest to the audio path.
        # Revert to the original path if the new path still doesn't exist.
        # Assume that the audio path is like "wavs/xxxxxx.wav".
        if item['audio_file'] is not None:
            item['audio_file'] = manifest.get_full_path(audio_file=item['audio_file'], manifest_file=manifest_file)

        # Visual file
        if 'image_filename' in item:
            item['visual_filepath'] = item.pop('image_filename')
        elif 'image_filepath' in item:
            item['visual_filepath'] = item.pop('image_filepath')
        elif 'video_filename' in item:
            item['visual_filepath'] = item.pop('video_filename')
        elif 'video_filepath' in item:
            item['visual_filepath'] = item.pop('video_filepath')
        elif 'visual_filepath' not in item:
            item['visual_filepath'] = None

        # if visual_filepath is a directory, make sure it exists
        # Otherwise, try prefix the relative path with the root dir of the manifest file
        if item['visual_filepath'] is not None and 
            os.path.isdir(item['visual_filepath']) and
            not os.path.exists(item['visual_filepath']):
            abs_path = os.path.join(os.path.dirname(manifest_file), item['visual_filepath'])
            if not os.path.exists(abs_path):
                item['visual_filepath'] = None
            else:
                item['visual_filepath'] = abs_path

        # Duration.
        if 'duration' not in item:
            item['duration'] = None

        # Answer.
        if self.answer_key in item:
            item['answer'] = item.pop(self.answer_key)
        elif 'text' in item:
            # compatability with ASR manifests that uses 'text' as answer key
            item['answer'] = item.pop('text')
        elif 'text_filepath' in item:
            with open(item.pop('text_filepath'), 'r') as f:
                item['answer'] = f.read()
        else:
            item['answer'] = "na"

        # context.
        if self.context_key in item:
            item['context'] = item.pop(self.context_key)
        elif 'context_filepath' in item:
            with open(item.pop('context_filepath'), 'r') as f:
                item['context'] = f.read()
        elif self.context_list is not None:
            context = np.random.choice(self.context_list).strip()
            item['context'] = context
        elif 'question' in item:
            # compatability with old manifests that uses 'question' as context key
            logging.warning(
                f"Neither `{self.context_key}` is found nor"
                f"`context_file` is set, but found `question` in item: {item}",
                mode=logging_mode.ONCE,
            )
            item['context'] = item.pop('question')
        else:
            # default context if nothing is found
            item['context'] = "what does this audio mean"

        item = dict(
            audio_file=item['audio_file'],
            visual_filepath=item['visual_filepath'],
            duration=item['duration'],
            context=str(item['context']),
            answer=str(item['answer']),
            offset=item.get('offset', None),
            speaker=item.get('speaker', None),
            orig_sr=item.get('orig_sample_rate', None),
            lang=item.get('lang', None),
        )
        return item

class WdsAudioVisualFilter:
    """
    filter function for tarred audio and visual files, skip entry if not in manifest
    """

    def __init__(self, collection, iterator):
        self.iterator = iterator
        self.collection = collection

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            audio_bytes, image_pixels, video_bytes, key = next(self.iterator)
            file_id, _ = os.path.splitext(os.path.basename(key))
            if file_id in self.collection.mapping:
                return audio_bytes, image_pixels, video_bytes, key
            else:
                logging.warning(f"key not in manifest: {file_id}", mode=logging_mode.ONCE)


class WdsAudioVisualLoopOffsets:
    """
    Loop over wds audio and visual files
    """

    def __init__(self, collection, iterator):
        self.iterator = iterator
        self.collection = collection
        self.current_fid = None
        self.current_audio_bytes = None
        self.current_image_pixels = None
        self.current_video_bytes = None
        self.offset_id = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_fn is None:
            self.current_audio_bytes, self.current_image_pixels, self.current_video_bytes, self.current_fid = \
                next(self.iterator)
            self.offset_id = 0
        else:
            offset_list = self.collection.mapping[self.current_fid]
            if len(offset_list) == self.offset_id + 1:
                self.current_audio_bytes, self.current_image_pixels, self.current_video_bytes, self.current_fid = \
                    next(self.iterator)
                self.offset_id = 0
            else:
                self.offset_id += 1

        return self.current_audio_bytes, self.current_image_pixels, self.current_video_bytes, self.current_fid, self.offset_id

class AudioVisualTextWebDataset(IterableDataset):
    """
    A Dataset which loads webDataset compliant audio and image/video files.

    Accepts a single comma-separated JSON manifest file containing paths to audio files, image/videos files, transcripts,
    and durations (in seconds).
    Each new line is a different sample. Example below:

    .. code-block:: json

        {"audio_filepath": "1.wav", "visual_filepath": "1/*.jpg", "duration": 1.12, "question": "what is the capital of France?", "answer": "Paris"}
        {"audio_filepath": "2.wav", "visual_filepath": "2/*.jpg", "duration": 2.15, "question": "what is the capital of Italy?", "answer": "Rome"}
    as well as the path(s) to the tarball(s) containing the wav, jpg/png/mp4 files. Each line of the manifest should
    contain the information for one audio file, one image/video file or path to sequence of images including at least the transcript and name of the audio
    and the image/video files within the tarball. 
    The visual_filepath can be of a path to a sequence of images belonging to a single sample or a single file:
    "visual_filepath": "1/*.jpg"
    "visual_filepath": "1.jpg"
    "visual_filepath": "1.mp4"
    ...

    Valid formats for the audio_visual_tar_filepaths argument include:
    (1) a single string that can be brace-expanded, e.g. 'path/to/audio_visual.tar' or 'path/to/audio_visual_{1..100}.tar.gz', or
    (2) a list of file paths that will not be brace-expanded, e.g. ['audio_visual_1.tar', 'audio_visual_2.tar', ...].

    Note: For brace expansion in (1), there may be cases where `{x..y}` syntax can't be used due to shell.
    This occurs most commonly inside SLURM scripts. Therefore we provide a few equivalent replacements.
    Supported opening braces - { <=> (, [, < and the special tag _OP_.
    Supported closing braces - } <=> ), ], > and the special tag _CL_.
    For SLURM based tasks, we suggest the use of the special tags for ease of use.

    See the WebDataset documentation for more information about accepted data and input formats.

    If using multiple workers the number of shards should be divisible by world_size to ensure an
    even split among workers. If it is not divisible, logging will give a warning but training will proceed.
    In addition, if using mutiprocessing, each shard MUST HAVE THE SAME NUMBER OF ENTRIES after filtering
    is applied. We currently do not check for this, but your program may hang if the shards are uneven!

    Additionally, please note that the len() of this DataLayer is assumed to be the length of the manifest
    after filtering. An incorrect manifest length may lead to some DataLoader issues down the line.

    Args:
        audio_visual_tar_filepaths: Either a list of audio tarball filepaths, or a
            string (can be brace-expandable).
        manifest_filepath (str): Path to the manifest.
        text_processor: TextProcessing object,
        sample_rate (int): Sample rate to resample loaded audio to
        int_values (bool): If true, load samples as 32-bit integers. Defauts to False.
        audio_augmentor (nemo.collections.asr.parts.perturb.AudioAugmentor): An AudioAugmentor
            object used to augment loaded audio
        shuffle_n (int): How many samples to look ahead and load to be shuffled.
            See WebDataset documentation for more details.
            Defaults to 0.
        min_duration (float): Dataset parameter.
            All training files which have a duration less than min_duration
            are dropped. Note: Duration is read from the manifest JSON.
            Defaults to 0.1.
        max_duration (float): Dataset parameter.
            All training files which have a duration more than max_duration
            are dropped. Note: Duration is read from the manifest JSON.
            Defaults to None.
        blank_index (int): Blank character index, defaults to -1.
        unk_index (int): Unknown character index, defaults to -1.
        normalize (bool): Dataset parameter.
            Whether to use automatic text cleaning.
            It is highly recommended to manually clean text for best results.
            Defaults to True.
        trim (bool): Whether to use trim silence from beginning and end
            of audio signal using librosa.effects.trim().
            Defaults to False.
        bos_id (id): Dataset parameter.
            Beginning of string symbol id used for seq2seq models.
            Defaults to None.
        eos_id (id): Dataset parameter.
            End of string symbol id used for seq2seq models.
            Defaults to None.
        pad_id (id): Token used to pad when collating samples in batches.
            If this is None, pads using 0s.
            Defaults to None.
        shard_strategy (str): Tarred dataset shard distribution strategy chosen as a
            str value during ddp.

            - `scatter`: The default shard strategy applied by WebDataset, where each node gets
              a unique set of shards, which are permanently pre-allocated and never changed at runtime.
            - `replicate`: Optional shard strategy, where each node gets all of the set of shards
              available in the tarred dataset, which are permanently pre-allocated and never changed at runtime.
              The benefit of replication is that it allows each node to sample data points from the entire
              dataset independently of other nodes, and reduces dependence on value of `shuffle_n`.

            :warning: Replicated strategy allows every node to sample the entire set of available tarfiles,
                and therefore more than one node may sample the same tarfile, and even sample the same
                data points! As such, there is no assured guarantee that all samples in the dataset will be
                sampled at least once during 1 epoch. Scattered strategy, on the other hand, on specific
                occasions (when the number of shards is not divisible with ``world_size``), will not sample
                the entire dataset. For these reasons it is not advisable to use tarred datasets as validation
                or test datasets.

        shard_manifests (bool): Whether or not to try / shard manifests. Defaults to False.
        global_rank (int): Worker rank, used for partitioning shards. Defaults to 0.
        world_size (int): Total number of processes, used for partitioning shards. Defaults to 0.

            :note: Below args are NLP-specific

        max_seq_length (int): maximum sequence length for each dataset examples. Examples will either be truncated
            to fit this length or dropped if they cannot be truncated.
        min_seq_length (int): min length of each data example in the dataset. Data examples will be dropped if they
            do not meet the min length requirements.
        tokens_to_generate (int): maximum tokens to generate in a single pass. Defaults to 128.
        context_key: Key to use for the context in your JSONL file
        answer_key: Key to use for the label in your JSONL file
        context_file: Optional[Union[List[str], str]] = None, if provided, will use this file to load
            random questions from, if question is not in manifest.
    """

    def __init__(
        self,
        audio_visual_tar_filepaths: Union[str, List[str]],
        manifest_filepath: str,
        text_processor: TextProcessing,
        visual_processor,
        sample_rate: int,
        int_values: bool = False,
        audio_augmentor: Optional['nemo.collections.asr.parts.perturb.AudioAugmentor'] = None,
        visual_augmentor = None,
        shuffle_n: int = 0,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
        trim: bool = False,
        shard_strategy: str = "scatter",
        shard_manifests: bool = False,
        global_rank: int = 0,
        world_size: int = 0,
        max_seq_length: int = 1024,
        min_seq_length: int = 1,
        tokens_to_generate: int = 128,
        pad_to_max_length: bool = False,
        context_key: str = 'context',
        answer_key: str = 'answer',
        context_file: Optional[Union[List[str], str]] = None,
    ):
        super().__init__()
        self.text_processor = text_processor
        self.visual_processor = visual_processor
        self.max_seq_length = max_seq_length
        self.min_seq_length = min_seq_length
        self.is_megatron_iterable = True
        self.shard_manifests = shard_manifests
        self.tokens_to_generate = tokens_to_generate
        self.pad_to_max_length = pad_to_max_length

        self.collection = AVLMAudioVisualTextCollection(
            manifests_files=manifest_filepath,
            min_duration=min_duration,
            max_duration=max_duration,
            index_by_file_id=True,
            context_file=context_file,
            context_key=context_key,
            answer_key=answer_key,
        )

        self.len = self._compute_len()

        self.waveform_featurizer = WaveformFeaturizer(sample_rate=sample_rate, int_values=int_values, audio_augmentor=audio_augmentor)
        self.trim = trim

        audio_visual_tar_filepaths = expand_sharded_filepaths(
            sharded_filepaths=audio_visual_tar_filepaths,
            shard_strategy=shard_strategy,
            world_size=world_size,
            global_rank=global_rank,
        )

        # Put together WebDataset
        self._dataset = wds.WebDataset(urls=audio_visual_tar_filepaths, nodesplitter=None)

        if shuffle_n == 0:
            logging.info("WebDataset will not shuffle files within the tar files.")

        # Put together WebDataset pipeline
        self._dataset = wds.DataPipeline(
            wds.SimpleShardList(urls=audio_visual_tar_filepaths),
            webdataset_split_by_workers,
            wds.shuffle(shuffle_n),
            wds.tarfile_to_samples(),
            wds.decode('pil'),
            wds.rename(
                audio=VALID_AUDIO_FILE_FORMATS, 
                image=';'.join(VALID_IMAGE_FILE_FORMATS_SET),
                video=';'.join(VALID_VIDEO_FILE_FORMATS_SET), 
                key='__key__'),
            wds.to_tuple('audio', 'image', 'video', 'key', missing_is_error=False),
            self._filter,
            self._loop_offsets,
            wds.map(self._build_sample),
        )

    def _filter(self, iterator):
        """This function is used to remove samples that have been filtered out by ASRAudioText already.
        Otherwise, we would get a KeyError as _build_sample attempts to find the manifest entry for a sample
        that was filtered out (e.g. for duration).
        Note that if using multi-GPU training, filtering may lead to an imbalance in samples in each shard,
        which may make your code hang as one process will finish before the other.
        """
        return WdsAudioVisualFilter(self.collection, iterator)

    def _loop_offsets(self, iterator):
        """This function is used to iterate through utterances with different offsets for each file."""
        return WdsAudioVisualLoopOffsets(self.collection, iterator)

    def _collate_fn(self, batch):
        # TODO
        return None

    def collate_fn(self, batch):
        # override collate_fn to skip type checking
        return self._collate_fn(batch)

    def _build_sample(self, tup):
        """Builds the training sample by combining the data from the WebDataset with the manifest info."""
        audio_bytes, image_pixels, video_bytes, key, offset_id = tup

        if key is not None:
            # Grab manifest entry from self.manifest_preprocessor.collection
            file_id, _ = os.path.splitext(os.path.basename(key))
            manifest_idx = self.collection.mapping[file_id][offset_id]
            manifest_entry = self.collection[manifest_idx]

            # init output dict
            output = {"idx": manifest_idx}

            offset = manifest_entry.offset
            if offset is None:
                offset = 0

            # process audio
            if audio_bytes is not None:                
                # Convert audio bytes to IO stream for processing (for SoundFile to read)
                audio_filestream = io.BytesIO(audio_bytes)
                audio_features = self.waveform_featurizer.process(
                    audio_filestream,
                    offset=offset,
                    duration=manifest_entry.duration,
                    trim=self.trim,
                    orig_sr=manifest_entry.orig_sr,
                )
                audio_filestream.close()

                # Audio features
                output["audio_signal"] = audio_features
                output["audio_length"] = torch.tensor(audio_features.shape[0]).long()
            else:
                # dummy audio_features
                output["audio_signal"] = torch.zeros([80])
                # accomodates normalize_batch
                output["audio_length"] = torch.tensor(80)

            # process image
            # TODO: dummy image output
            if image_pixels is not None:
                # convert to torch tensor
                image_pixels = torchvision.transforms.functional.pil_to_tensor(image_pixels)
                
                # process image
                if self.visual_processor is not None:
                    output["visual_signal"] = self.visual_processor(image_pixels[i])
                else:
                    output["visual_signal"] = image_pixels.unsqueeze(0)

                output["num_media_tiles"] = output["visual_signal"].shape[0]
                height = image_pixels.shape[1]
                width = image_pixels.shape[2]
                output["image_sizes"] = torch.tensor([[height, width]], dtype=torch.long)

            # TODO: process video. For videos we have to read the raw bytes and deocde it here since 
            # we need to know the offset and the duration

        # Text features
        text_data = self.text_processor(context=manifest_entry.context, output=manifest_entry.answer)

        output.update(text_data)

        if image_pixels is not None or video_bytes is not None:
            output["attention_mask"] = torch.ones(len(text_data), dtype=torch.long)

        output['metadata'] = {
            'audio_filepath': manifest_entry.audio_file,
            'visual_filepaths': manifest_entry.visual_files,
            'offset': offset,
            'duration': manifest_entry.duration,
        }
        return output

    def get_manifest_sample(self, sample_id):
        """
        return manifest item given the index
        """
        return self.collection[sample_id]

    def __iter__(self):
        return self._dataset.__iter__()

    def _compute_len(self):
        # TODO: need to figure out why here needs to be divided by world_size, while in ASR we don't need to.
        if self.shard_manifests and torch.distributed.is_available() and torch.distributed.is_initialized():
            my_len = torch.tensor(len(self.collection), dtype=torch.int32).cuda()
            torch.distributed.all_reduce(my_len)
            my_len = my_len.int() // parallel_state.get_data_parallel_world_size()
            logging.info(f'Sharded manifests: Total length: {my_len}')
        else:
            my_len = len(self.collection) // parallel_state.get_data_parallel_world_size()

        return my_len

    def __len__(self):
        return self.len

def get_audio_visual_text_webdataset_from_config(
    config: DictConfig,
    text_processor: TextProcessing,
    visual_processor,
    audio_augmentor,
    visual_augmentor,
    global_rank: int = 0,
    world_size: int = 1,
):
    """
    Get tarred dataset from config
    """
    # TODO: to be implemented
    return None
