import json
import math
import multiprocessing
from collections import Counter
from itertools import repeat

import torch
from tqdm import tqdm


def gen_binary_pure(data, args):
    unit = args['unit']
    duration = data['duration']
    offset = data['offset'] if 'offset' in data else 0
    zeros = math.floor(duration / unit)


def gen_binary_seq_speech(data, args):
    oracle = False  # rttm ctm issue
    unit = args['unit']
    duration = data['duration']
    if 'energy_vad' in data:
        speech_segments = torch.load(data['energy_vad'])
    else:
        speech_segments = torch.load(data['oracle_vad'])
        oracle = True

    offset = data['offset'] if 'offset' in data else 0

    out = []
    for i in range(len(speech_segments)):
        if i == 0:
            zeros = float((speech_segments[i][0] - offset) / unit)
        else:
            zeros = float((speech_segments[i][0] - speech_segments[i - 1][1]) / unit)

        out.extend([0] * round(zeros))  # math.floor

        ones = float((speech_segments[i][1] - speech_segments[i][0]) / unit)
        out.extend([1] * round(ones))

    if not oracle:  # duration may be longer than annotated data.
        last_zeros = float((duration - speech_segments[i][1]) / unit)
        out.extend([0] * round(last_zeros))
    else:
        rttm_duration = float(speech_segments[i][1])
        #         if  data['duration']< 599:
        #             print("old",  data['duration'], "new", rttm_duration )
        data['duration'] = rttm_duration

    silence_ratio = Counter(out)[0] / (Counter(out)[0] + Counter(out)[1])

    out_str = ' '.join(str(x) for x in out)
    if 'energy_vad' in data:
        data['energy_vad_mask'] = out_str
    else:
        data['oracle_vad_mask'] = out_str

    data['silence_ratio'] = silence_ratio

    return data


def gen_binary_seq_speech_star(args):
    """
    A workaround for tqdm with starmap of multiprocessing
    """
    return gen_binary_seq_speech(*args)


def main():

    #     asr_manifest = "/data/russian/train_energy.json"
    #     asr_manifest = "/data/spanish/mls/train_energy.json"
    #     asr_manifest =  "/data/german/mcv/train_energy.json"
    #     asr_manifest = "/data/french/mls/train_energy.json"
    #     asr_manifest = "/data/mandarin/aishell2/dev_energy.json"

    #     asr_manifest = "/home/fjia/code/0_data_vad_sd/sd_oracle_vad_manifest/fisher_2005_oracle.json"
    asr_manifest = "/home/fjia/code/0_data_vad_sd/sd_oracle_vad_manifest/ami_dev_oracle.json"
    #     asr_manifest = "/home/fjia/code/0_data_vad_sd/sd_oracle_vad_manifest/ch120_moved_oracle.json"

    out_manifest = "binary_manifests/ami_dev_10ms.json"

    unit = 0.01  # 40ms
    number_of_processes = 20

    input_data = []
    for line in open(asr_manifest, 'r'):
        input_data.append(json.loads(line))

    args_func = {'unit': unit}

    inputs = zip(input_data, repeat(args_func))

    with multiprocessing.Pool(number_of_processes) as p:
        results = []
        results = list(tqdm(p.imap(gen_binary_seq_speech_star, inputs), total=len(input_data)))

    with open(out_manifest, "w") as fout:
        for result in results:
            json.dump(result, fout)
            fout.write('\n')
            fout.flush()


if __name__ == '__main__':
    main()
