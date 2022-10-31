import glob
import json
import multiprocessing
import random

import IPython.display as ipd
import librosa
import torch
from tqdm import tqdm
from unsupervised_vad import *

from nemo.collections.asr.parts.utils.vad_utils import *


def cal_dur(speech_segments):
    dur = 0
    for i in speech_segments:
        dur += i[1] - i[0]
    return dur


def to_start_end(vad, min_duration_on=1):
    #     speech_segments = set()
    speech_segments = torch.empty(0)

    start = 0
    for i in range(0, len(vad) - 1):
        # 1111000
        if vad[i] == 1 and vad[i + 1] == 0:
            end = i
            if end / 16000 - start / 16000 > min_duration_on:
                new_seg = torch.tensor([start / 16000, end / 16000]).unsqueeze(0)
                #                 speech_segments.add((start/16000, end/16000))
                speech_segments = torch.cat((speech_segments, new_seg), 0)
        # 0001111
        if vad[i] == 0 and vad[i + 1] == 1:
            start = i + 1

    if vad[-1] == 1:
        end = len(vad) - 1
        if end / 16000 - start / 16000 > min_duration_on:
            #             speech_segments.add((start/16000, end/16000))
            new_seg = torch.tensor([start / 16000, end / 16000]).unsqueeze(0)
            speech_segments = torch.cat((speech_segments, new_seg), 0)
    return speech_segments


def process_one_file(a, args_func):
    speech_manifest = False  # todo
    offset = 0
    dur = a['duration']

    audio_file = a['audio_filepath']
    audio, sample_rate = read_wav(audio_file, offset=offset, duration=dur, sample_rate=16000,)

    win_len = int(sample_rate * 0.025)
    hop_len = int(sample_rate * 0.010)
    percent_high_nrg = 0.01
    per_args = {"min_duration_on": 0.1, "min_duration_off": 0.5}  # french # mandarin # spanish # german # russian

    audio_filename = audio_file.split("/")[-1].split(".wav")[0]

    sframes = enframe(audio, win_len, hop_len)  # rows: frame index, cols: each frame
    _, vad = nrg_vad(sframes, percent_high_nrg)
    vad_decision = deframe(vad, win_len, hop_len)
    vad_decision = vad_decision.squeeze()
    speech_segments = to_start_end(vad_decision, per_args['min_duration_on'])
    speech_segments = filtering(speech_segments, per_args)
    #     speech_dur = cal_dur(speech_segments)

    if args_func['to_speech_manifest']:
        metas = []
        for i in speech_segments:
            offset = i[0]
            duration = i[1] - i[0]
            metadata = {
                'audio_filepath': audio_file,
                'offset': round(offset, 4),
                'duration': round(duration, 4),
                'label': "speech",
                'text': '_',  # Not accurate for speech segments and not useful
            }
            metas.append(metadata)
        return metas

    else:
        energy_vad_tensor_path = os.path.join(args_func['energy_vad_folder'], audio_filename + '.pt')
        torch.save(speech_segments, energy_vad_tensor_path)

        a['energy_vad'] = energy_vad_tensor_path
        return a


def process_one_file_star(args):
    """
    A workaround for tqdm with starmap of multiprocessing
    """
    return process_one_file(*args)


def main():
    """
    # german
    output_path =  "/data/german/mcv/mcv7.0_train_manifest_cleaned_speech_new.json"
    asr_manifest = "/data/german/mcv/mcv7.0_train_manifest_cleaned.json"
    
    
    # french
    output_path = "/data/french/mls/final_test_speech.json"
    asr_manifest = "/data/french/mls/final_test.json"
    
    # mandarin
    output_path = "/data/mandarin/aishell2/train_speech.json"
    asr_manifest = "/data/mandarin/aishell2/train.json"
    
    # spanish
    output_path = "/data/spanish/mls/dev_speech.json"
    asr_manifest = "/data/spanish/mls/dev.json"    
    
    
    # russian
    output_path = "/data/russian/mcv_ruls_dev_speech.json"
    asr_manifest = "/data/russian/mcv_ruls_dev.json"
    
    ###############
    """

    subset = "test"
    """
    # german dev 26.333886685000014 train 567.384683333323 test 26.592823576388874
    output_path =  f"/data/german/mcv/{subset}_energy.json"
    asr_manifest = f"/data/german/mcv/mcv7.0_{subset}_manifest_cleaned.json"
    energy_vad_folder = f"/data/german/mcv/{subset}_energy_vad"
    
    
    # french dev 9.375104340555566 train 1036.090349530815 test 9.636269496388895
    output_path = f"/data/french/mls/{subset}_energy.json"
    asr_manifest = f"/data/french/mls/final_{subset}.json"
    energy_vad_folder = f"/data/french/mls/{subset}_energy_vad"
    
    
    # mandarin dev 2.028138697916663 train 1000.2204885416209 test 4.004922013888893
    output_path = f"/data/mandarin/aishell2/{subset}_energy.json"
    asr_manifest = f"/data/mandarin/aishell2/{subset}.json"
    energy_vad_folder = f"/data/mandarin/aishell2/{subset}_energy_vad"
    
   
    # spanish dev 9.98866147611112 train 917.6841761097153 test 10.003648993055572
    output_path = f"/data/spanish/mls/{subset}_energy.json"
    asr_manifest = f"/data/spanish/mls/{subset}.json"    
    energy_vad_folder = f"/data/spanish/mls/{subset}_energy_vad"
   
    
    # russian dev 15.216641666666675 train 116.36067777778014 test 15.932772222222187
    output_path = f"/data/russian/{subset}_energy.json"
    asr_manifest = f"/data/russian/mcv_ruls_{subset}.json"
    energy_vad_folder = f"/data/russian/{subset}_energy_vad"
   
    """
    data = []
    for line in open(asr_manifest, 'r'):
        data.append(json.loads(line))

    org_duration = 0
    for i in range(len(data)):
        duration = data[i]['duration']
        org_duration += duration

    print(f"{org_duration / 3600} hours")

    #     sample_data = data[:100]
    #     data = sample_data

    number_of_processes = 15
    to_speech_manifest = False

    args_func = {'energy_vad_folder': energy_vad_folder, 'to_speech_manifest': to_speech_manifest}
    if not os.path.exists(energy_vad_folder):
        os.mkdir(energy_vad_folder)

    p = multiprocessing.Pool(number_of_processes)
    results = []

    inputs = zip(data, repeat(args_func))

    #     for result in tqdm.tqdm(p.imap_unordered(process_one_file, data), total=len(data)):
    #         results.append(result)
    results = list(tqdm(p.imap(process_one_file_star, inputs), total=len(data)))
    p.close()

    if to_speech_manifest:

        with open(output_path, "w") as fout:
            for result in results:
                # each file might have multi meta
                for meta in result:
                    json.dump(meta, fout)
                    fout.write('\n')
                    fout.flush()
    else:
        with open(output_path, "w") as fout:
            for result in results:
                json.dump(result, fout)
                fout.write('\n')
                fout.flush()


if __name__ == '__main__':
    main()
