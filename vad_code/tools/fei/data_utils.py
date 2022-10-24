import json
import os

import numpy as np
import pandas as pd


def construct_speech_segment(rttm_path):
    data = pd.read_csv(rttm_path, sep=" ", delimiter=None, header=None)
    data = data.rename(columns={3: "start", 4: "dur", 7: "speaker"})

    data['start'] = data['start'].astype(float)
    data['dur'] = data['dur'].astype(float)
    data['end'] = data['start'] + data['dur']

    data = data.sort_values(by=['start'])
    data['segment'] = list(zip(data['start'], data['end']))

    return data


def merge(times):
    # TODO: add overlap detection
    saved = list(times[0])
    for st, en in sorted([sorted(t) for t in times]):
        if st <= saved[1]:
            saved[1] = max(saved[1], en)
        else:
            yield tuple(saved)
            saved[0] = st
            saved[1] = en
    yield tuple(saved)


def gen_df_all_long(data):

    merged_segments = list(merge(list(data['segment'])))
    df_speech = pd.DataFrame(merged_segments, columns=['start', 'end'])
    df_speech['dur'] = df_speech['end'] - df_speech['start']
    df_speech['label'] = 'speech'
    df_speech.head()

    df_all = pd.DataFrame(columns=['start', 'dur', 'end', 'label'])
    for index, row in df_speech.iterrows():
        # add speech line
        df_all = df_all.append(
            {
                'start': df_speech['start'][index],
                'dur': df_speech['dur'][index],
                'end': df_speech['start'][index] + df_speech['dur'][index],
                'label': df_speech['label'][index],
            },
            ignore_index=True,
        )
        if index < len(df_speech) - 1:
            end = df_speech['start'][index] + df_speech['dur'][index]
            # add non-speech line
            if end < df_speech['start'][index + 1]:
                df_all = df_all.append(
                    {
                        'start': end,
                        'dur': df_speech['start'][index + 1] - end,
                        'label': 'background',
                        'end': df_speech['start'][index + 1],
                    },
                    ignore_index=True,
                )

    print(f" Number of generated rows: {len(df_all)}")
    df_all_long = df_all[df_all["dur"] >= 0.63]
    print(f" Number of long enough rows: {len(df_all_long)}")
    return df_all_long


def gen_manifest(name, audio_path, seg_len, df_all_long, json_folder):
    offsets = []
    durations = []
    labels = []

    for index, row in df_all_long.iterrows():
        dur = row["dur"]
        i = 0
        while dur >= seg_len:
            offsets.append(row['start'] + i * seg_len)
            labels.append(row['label'])
            durations.append(seg_len)
            i += 1
            dur -= seg_len

    labels = np.array(labels)

    json_folder = json_folder + "_" + str(seg_len)
    if not os.path.exists(json_folder):
        print(f"Creating {json_folder}")
        os.mkdir(json_folder)

    json_file = os.path.join(json_folder, name + ".json")

    with open(json_file, 'w') as fout:
        for duration, offset, label in zip(durations, offsets, labels):
            metadata = {
                'audio_filepath': audio_path,
                'duration': duration,
                'label': label,
                'text': '_',  # for compatibility with ASRAudioText
                'offset': offset,
            }
            json.dump(metadata, fout)
            fout.write('\n')
            fout.flush()

    seg_b = len(labels[labels == 'background'])
    seg_s = len(labels[labels == 'speech'])

    return seg_b, seg_s
