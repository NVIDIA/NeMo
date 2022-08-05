# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import json
import os
from collections import Counter
from collections import OrderedDict as od

import numpy as np

from nemo.collections.asr.parts.utils.speaker_utils import (
    audio_rttm_map,
    get_subsegments,
    get_uniqname_from_filepath,
    rttm_to_labels,
)


def rreplace(s, old, new):
    """
    Replace end of string.

    Args:
        s (str): string to operate on
        old (str): ending of string to replace
        new (str): replacement for ending of string
    Returns:
        new.join(li) (string): new string with end replaced
    """
    li = s.rsplit(old, 1)
    return new.join(li)


def get_uniq_id_with_period(path):
    """
    Get file id.

    Args:
        path (str): path to audio file
    Returns:
        uniq_id (string): unique speaker ID
    """
    split_path = os.path.basename(path).split('.')[:-1]
    uniq_id = '.'.join(split_path) if len(split_path) > 1 else split_path[0]
    return uniq_id


def get_subsegment_dict(subsegments_manifest_file, window, shift, deci):
    """
    Get subsegment dictionary from manifest file.

    Args:
        subsegments_manifest_file (str): Path to subsegment manifest file
        window (float): Window length for segmentation
        shift (float): Shift length for segmentation
        deci (int): Rounding number of decimal places
    Returns:
        _subsegment_dict (_subsegment_dict): Subsegment dictionary
    """
    _subsegment_dict = {}
    with open(subsegments_manifest_file, 'r') as subsegments_manifest:
        segments = subsegments_manifest.readlines()
        for segment in segments:
            segment = segment.strip()
            dic = json.loads(segment)
            audio, offset, duration, label = dic['audio_filepath'], dic['offset'], dic['duration'], dic['label']
            subsegments = get_subsegments(offset=offset, window=window, shift=shift, duration=duration)
            uniq_id = get_uniq_id_with_period(audio)
            if uniq_id not in _subsegment_dict:
                _subsegment_dict[uniq_id] = {'ts': [], 'json_dic': []}
            for subsegment in subsegments:
                start, dur = subsegment
            _subsegment_dict[uniq_id]['ts'].append([round(start, deci), round(start + dur, deci)])
            _subsegment_dict[uniq_id]['json_dic'].append(dic)
    return _subsegment_dict


def get_input_manifest_dict(input_manifest_path):
    """
    Get dictionary from manifest file.

    Args:
        input_manifest_path (str): Path to manifest file
    Returns:
        input_manifest_dict (dict): Dictionary from manifest file
    """
    input_manifest_dict = {}
    with open(input_manifest_path, 'r') as input_manifest_fp:
        json_lines = input_manifest_fp.readlines()
        for json_line in json_lines:
            dic = json.loads(json_line)
            dic["text"] = "-"
            uniq_id = get_uniqname_from_filepath(dic["audio_filepath"])
            input_manifest_dict[uniq_id] = dic
    return input_manifest_dict


def write_truncated_subsegments(input_manifest_dict, _subsegment_dict, output_manifest_path, step_count, deci):
    """
    Write subsegments to manifest filepath.

    Args:
        input_manifest_dict (dict): Input manifest dictionary
        _subsegment_dict (dict): Input subsegment dictionary
        output_manifest_path (str): Path to output manifest file
        step_count (int): Number of the unit segments you want to create per utterance
        deci (int): Rounding number of decimal places
    """
    with open(output_manifest_path, 'w') as output_manifest_fp:
        for uniq_id, subseg_dict in _subsegment_dict.items():
            # print(f"Writing {uniq_id}")
            subseg_array = np.array(subseg_dict['ts'])
            subseg_array_idx = np.argsort(subseg_array, axis=0)
            chunked_set_count = subseg_array_idx.shape[0] // step_count

            for idx in range(chunked_set_count - 1):
                chunk_index_stt = subseg_array_idx[:, 0][idx * step_count]
                chunk_index_end = subseg_array_idx[:, 1][(idx + 1) * step_count]
                offset_sec = subseg_array[chunk_index_stt, 0]
                end_sec = subseg_array[chunk_index_end, 1]
                dur = round(end_sec - offset_sec, deci)
                meta = input_manifest_dict[uniq_id]
                meta['offset'] = offset_sec
                meta['duration'] = dur
                json.dump(meta, output_manifest_fp)
                output_manifest_fp.write("\n")


def write_file(name, lines, idx):
    """
    Write json lines to file.

    Args:
        name (str): Output file path
        lines (list): List of json lines
        idx (int): Indices to dump to the file
    """
    with open(name, 'w') as fout:
        for i in idx:
            dic = lines[i]
            json.dump(dic, fout)
            fout.write('\n')


def read_file(pathlist):
    """
    Read list of lines from target file.

    Args:
        pathlist (str): Input file path
    Returns:
        sorted(pathlist) (list): List of lines
    """
    pathlist = open(pathlist, 'r').readlines()
    return sorted(pathlist)


def get_dict_from_wavlist(pathlist):
    """
    Read dictionaries from list of lines

    Args:
        pathlist (list): List of file paths
    Returns:
        path_dict (dict): Dictionary containing dictionaries read from files
    """
    path_dict = od()
    pathlist = sorted(pathlist)
    for line_path in pathlist:
        uniq_id = os.path.basename(line_path).split('.')[0]
        path_dict[uniq_id] = line_path
    return path_dict


def get_dict_from_list(data_pathlist, uniqids):
    """
    Create dictionaries from list of lines

    Args:
        data_pathlist (list): List of file paths
        uniqids (list): List of file IDs
    Returns:
        path_dict (dict): Dictionary containing file paths
    """
    path_dict = {}
    for line_path in data_pathlist:
        uniq_id = os.path.basename(line_path).split('.')[0]
        if uniq_id in uniqids:
            path_dict[uniq_id] = line_path
        else:
            raise ValueError(f'uniq id {uniq_id} is not in wav filelist')
    return path_dict


def get_path_dict(data_path, uniqids, len_wavs=None):
    """
    Create dictionary from list of lines (using the get_dict_from_list function)

    Args:
        data_path (str): Path to file containing list of files
        uniqids (list): List of file IDs
        len_wavs (int): Length of file list
    Returns:
        data_pathdict (dict): Dictionary containing file paths
    """
    if data_path is not None:
        data_pathlist = read_file(data_path)
        if len_wavs is not None:
            assert len(data_pathlist) == len_wavs
            data_pathdict = get_dict_from_list(data_pathlist, uniqids)
    elif len_wavs is not None:
        data_pathdict = {uniq_id: None for uniq_id in uniqids}
    return data_pathdict


def create_segment_manifest(input_manifest_path, output_manifest_path, window, shift, step_count, deci):
    """
    Create segmented manifest file from base manifest file

    Args:
        input_manifest_path (str): Path to input manifest file
        output_manifest_path (str): Path to output manifest file
        window (float): Window length for segmentation
        shift (float): Shift length for segmentation
        step_count (int): Number of the unit segments you want to create per utterance
        deci (int): Rounding number of decimal places
    """
    if '.json' not in input_manifest_path:
        raise ValueError("input_manifest_path file should be .json file format")
    if output_manifest_path and '.json' not in output_manifest_path:
        raise ValueError("output_manifest_path file should be .json file format")
    elif not output_manifest_path:
        output_manifest_path = rreplace(input_manifest_path, '.json', f'_{step_count}seg.json')

    input_manifest_dict = get_input_manifest_dict(input_manifest_path)
    segment_manifest_path = rreplace(input_manifest_path, '.json', '_seg.json')
    subsegment_manifest_path = rreplace(input_manifest_path, '.json', '_subseg.json')
    min_subsegment_duration = 0.05
    step_count = int(step_count)

    input_manifest_file = open(input_manifest_path, 'r').readlines()
    input_manifest_file = sorted(input_manifest_file)
    AUDIO_RTTM_MAP = audio_rttm_map(input_manifest_path)
    segments_manifest_file = write_rttm2manifest(AUDIO_RTTM_MAP, segment_manifest_path, deci)
    # print(segments_manifest_file)
    subsegments_manifest_file = subsegment_manifest_path
    segments_manifest_to_subsegments_manifest(
        segments_manifest_file, subsegments_manifest_file, window, shift, min_subsegment_duration,
    )
    subsegments_dict = get_subsegment_dict(subsegments_manifest_file, window, shift, deci)
    write_truncated_subsegments(input_manifest_dict, subsegments_dict, output_manifest_path, step_count, deci)
    os.remove(segment_manifest_path)
    os.remove(subsegment_manifest_path)


def create_manifest(
    wav_path, manifest_filepath, text_path=None, rttm_path=None, uem_path=None, ctm_path=None, add_duration=False
):
    """
    Create base manifest file

    Args:
        wav_path (str): Path to list of wav files
        manifest_filepath (str): Path to output manifest file
        text_path (str): Path to list of text files
        rttm_path (str): Path to list of rttm files
        uem_path (str): Path to list of uem files
        ctm_path (str): Path to list of ctm files
    """
    if os.path.exists(manifest_filepath):
        os.remove(manifest_filepath)
    wav_pathlist = read_file(wav_path)
    wav_pathdict = get_dict_from_wavlist(wav_pathlist)
    len_wavs = len(wav_pathlist)
    uniqids = sorted(wav_pathdict.keys())

    text_pathdict = get_path_dict(text_path, uniqids, len_wavs)
    rttm_pathdict = get_path_dict(rttm_path, uniqids, len_wavs)
    uem_pathdict = get_path_dict(uem_path, uniqids, len_wavs)
    ctm_pathdict = get_path_dict(ctm_path, uniqids, len_wavs)

    lines = []
    for uid in uniqids:
        wav, text, rttm, uem, ctm = (
            wav_pathdict[uid],
            text_pathdict[uid],
            rttm_pathdict[uid],
            uem_pathdict[uid],
            ctm_pathdict[uid],
        )

        audio_line = wav.strip()
        if rttm is not None:
            rttm = rttm.strip()
            labels = rttm_to_labels(rttm)
            num_speakers = Counter([l.split()[-1] for l in labels]).keys().__len__()
        else:
            num_speakers = None

        if uem is not None:
            uem = uem.strip()

        if text is not None:
            text = open(text.strip()).readlines()[0].strip()
        else:
            text = "-"

        if ctm is not None:
            ctm = ctm.strip()

        duration = None
        if add_duration:
            y, sr = librosa.load(audio_line, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)
        meta = [
            {
                "audio_filepath": audio_line,
                "offset": 0,
                "duration": duration,
                "label": "infer",
                "text": text,
                "num_speakers": num_speakers,
                "rttm_filepath": rttm,
                "uem_filepath": uem,
                "ctm_filepath": ctm,
            }
        ]
        lines.extend(meta)

    write_file(manifest_filepath, lines, range(len(lines)))


def read_manifest(manifest):
    """
    Read manifest file

    Args:
        manifest (str): Path to manifest file
    Returns:
        data (list): List of JSON items
    """
    data = []
    with open(manifest, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            data.append(item)
    return data


def write_manifest(output_path, target_manifest):
    """
    Write to manifest file

    Args:
        output_path (str): Path to output manifest file
        target_manifest (list): List of manifest file entries
    """
    with open(output_path, "w") as outfile:
        for tgt in target_manifest:
            json.dump(tgt, outfile)
            outfile.write('\n')


def write_ctm(output_path, target_ctm):
    """
    Write ctm entries from diarization session to a .ctm file.

    Args:
        output_path (str): target file path
        target_ctm (dict): list of ctm entries
    """
    target_ctm.sort(key=lambda y: y[0])
    with open(output_path, "w") as outfile:
        for pair in target_ctm:
            tgt = pair[1]
            outfile.write(tgt)


def write_text(output_path, target_ctm):
    """
    Write text from diarization session to a .txt file

    Args:
        output_path (str): target file path
        target_ctm (dict): list of ctm entries
    """
    target_ctm.sort(key=lambda y: y[0])
    with open(output_path, "w") as outfile:
        for pair in target_ctm:
            tgt = pair[1]
            word = tgt.split(' ')[4]
            outfile.write(word + ' ')
        outfile.write('\n')


def write_rttm2manifest(AUDIO_RTTM_MAP: str, manifest_file: str, include_uniq_id: bool = False, deci: int = 5) -> str:
    """
    Write manifest file based on rttm files (or vad table out files). This manifest file would be used by
    speaker diarizer to compute embeddings and cluster them. This function takes care of overlapping VAD timestamps
    and trimmed with the given offset and duration value.

    Args:
        AUDIO_RTTM_MAP (dict):
            Dictionary containing keys to uniqnames, that contains audio filepath and rttm_filepath as its contents,
            these are used to extract oracle vad timestamps.
        manifest (str):
            The path to the output manifest file.

    Returns:
        manifest (str):
            The path to the output manifest file.
    """

    with open(manifest_file, 'w') as outfile:
        for uniq_id in AUDIO_RTTM_MAP:
            rttm_file_path = AUDIO_RTTM_MAP[uniq_id]['rttm_filepath']
            rttm_lines = read_rttm_lines(rttm_file_path)
            offset, duration = get_offset_and_duration(AUDIO_RTTM_MAP, uniq_id, deci)
            # print('offset: ', offset)
            # print('duration: ', duration)
            vad_start_end_list_raw = []
            for line in rttm_lines:
                start, dur = get_vad_out_from_rttm_line(line)
                # if start + dur > duration: #COLEMAN CHANGE HERE
                #    print('start+dur: ', start+dur)
                # else:
                vad_start_end_list_raw.append([start, start + dur])
            vad_start_end_list = combine_float_overlaps(vad_start_end_list_raw, deci)
            if len(vad_start_end_list) == 0:
                logging.warning(f"File ID: {uniq_id}: The VAD label is not containing any speech segments.")
            elif duration <= 0:
                logging.warning(f"File ID: {uniq_id}: The audio file has negative or zero duration.")
            else:
                overlap_range_list = getSubRangeList(
                    source_range_list=vad_start_end_list, target_range=[offset, offset + duration]
                )
                write_overlap_segments(outfile, AUDIO_RTTM_MAP, uniq_id, overlap_range_list, include_uniq_id, deci)
    return manifest_file


def segments_manifest_to_subsegments_manifest(
    segments_manifest_file: str,
    subsegments_manifest_file: str = None,
    window: float = 1.5,
    shift: float = 0.75,
    min_subsegment_duration: float = 0.05,
    include_uniq_id: bool = False,
):
    """
    Generate subsegments manifest from segments manifest file
    Args:
        segments_manifest file (str): path to segments manifest file, typically from VAD output
        subsegments_manifest_file (str): path to output subsegments manifest file (default (None) : writes to current working directory)
        window (float): window length for segments to subsegments length
        shift (float): hop length for subsegments shift
        min_subsegments_duration (float): exclude subsegments smaller than this duration value

    Returns:
        returns path to subsegment manifest file
    """
    # print(segments_manifest_file)
    # print(subsegments_manifest_file)
    if subsegments_manifest_file is None:
        pwd = os.getcwd()
        subsegments_manifest_file = os.path.join(pwd, 'subsegments.json')

    with open(segments_manifest_file, 'r') as segments_manifest, open(
        subsegments_manifest_file, 'w'
    ) as subsegments_manifest:
        segments = segments_manifest.readlines()
        for segment in segments:
            # print(segment)
            segment = segment.strip()
            dic = json.loads(segment)
            audio, offset, duration, label = dic['audio_filepath'], dic['offset'], dic['duration'], dic['label']
            subsegments = get_subsegments(offset=offset, window=window, shift=shift, duration=duration)
            if include_uniq_id and 'uniq_id' in dic:
                uniq_id = dic['uniq_id']
            else:
                uniq_id = None
            for subsegment in subsegments:
                start, dur = subsegment
                if dur > min_subsegment_duration:
                    meta = {
                        "audio_filepath": audio,
                        "offset": start,
                        "duration": dur,
                        "label": label,
                        "uniq_id": uniq_id,
                    }

                    json.dump(meta, subsegments_manifest)
                    subsegments_manifest.write("\n")

    return subsegments_manifest_file
