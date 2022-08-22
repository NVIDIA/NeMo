import json
import os
import shutil
from typing import Tuple
import time

import torch
from pyannote.core import Annotation, Segment
from pyannote.metrics import detection

from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.collections.asr.parts.utils.vad_utils import construct_manifest_eval, plot, stitch_segmented_asr_output, write_ss2manifest

from extract_speech import process_one_file
import multiprocessing
import tqdm
import numpy as np

from eval_vad_asr_utils import *



def main():
    """
    modes = ['streaming', 'offline']
    langs = ['english', 'mandarin', 'french', 'german',  'spanish', 'russian']
    vad_exps = ["novad", "oracle_vad", "neural_vad", "energy_vad", "random_vad"] 
    models = ['citrinet', 'nr_citrinet', 'nr_conformer_ctc', 'nr_conformer_transducer', 'nr_contextnet'] 
    db_list = [0,5,10,15,20,'clean']
    """

    db_list = ['clean'] 
    modes = ['offline'] #streaming
    langs = [
        # 'riva_silence',
        # 'driveix', 
        'chris', 
        # 'hub5',
        # 'GTC2019-keynote-parts', 
        # 'tmobile',
        # 'Ring_central_with_duration', #?
        # 'logmein',
        # 'StreamingInTheCloud',
        # 'tedlium2_test',
        # 'wsj-eval-92',
        # 'librivox-test-other',
        # 'yt_med_eval'
    ]

    vad_exps = ["novad"] 
    models = ['riva_conformer_ctc'] 

    shift_length_in_sec = 0.08
    clean_text = True

    num_workers = 1
    # ref="energy_vad"
    overlap=0.875
    smoothing=False
    ref='oracle_vad'
    random_vad_ref = 'energy_vad'

    subset="test"
    single= False # True
    exp = "_single" if single else ""
    exp = "_min10"
    res_file = f"res_asr_{exp}_offlines_s8_tunedO_test_new_mask.csv"

    si_ratio = False  #True

    fixed_silence_set = {1}
    if si_ratio:
        fixed_silence_set = set()
        for i in range(0, 11, 2):
            for j in range(0, 11, 2):
                fixed_silence_set.add((i,j))

    final_output_folder = f"asr_{exp}_s8_tuned_test_new_mask"

    # final_output_folder = "final_tuned"
    save_neural_vad = True
    os.makedirs(final_output_folder, exist_ok=True)

    for mode in modes:
        mode_folder = f'{final_output_folder}/{mode}'
        os.makedirs(mode_folder, exist_ok=True)

        for lang in langs:
            for model in models:
                if mode == "streaming":
                    if switch_model_buffered_param(model):
                        model_stride = switch_model_buffered_param(model)
                    else:
                        print(f"Currently do not support {model} in streaming/buffered model")
                        continue

                if switch_lang_model("english", model):
                    use_cer, no_space, use_model_path, asr_model = switch_lang_model("english", model)
                else:
                    print(f"{lang} with {model} does not exist")
                    continue

                for vad_exp in vad_exps:
                    for db in db_list:

                        for fixed_silence in fixed_silence_set:
                            start = time.time()
                            mode_lang_folder = f"{final_output_folder}/{mode}/{lang}"
                            os.makedirs(mode_lang_folder, exist_ok=True)

                            if db=='clean' :
                                # input_manifest=f"/home/fjia/code/5_syn/{lang}_{subset}{exp}.json"
                                # input_manifest=f"/home/fjia/code/5_syn/{lang}_{subset}{exp}.json"
                                # input_manifest = f"/home/fjia/code/asr_eval_manifests/{lang}.json" # todo
                                input_manifest =f"vad_ss_output/{lang}.json"
                            else:
                                # silence only clean now need change
                                # input_manifest = f"/data/syn_noise_augmented/manifests/{lang}_{subset}{exp}_test_noise_0_30_musan_fs_{db}db.json"
                                # input_manifest = f"/data/syn_noise_augmented_fixed/manifests/{lang}_{subset}{exp}_test_noise_0_30_musan_fs_{db}db.json"
                                # input_manifest = f"/data/syn_noise_augmented_fixed_10_100/manifests/{lang}_{subset}{exp}_test_noise_10_100_musan_fs_{db}db.json"
                                print("error")
                                raise 

                            if mode == "offline":
                                if vad_exp =="novad":
                                    novad_output_manifest= f"{final_output_folder}/{mode}/{lang}/asr_{vad_exp}_{model}_output_manifest_{db}_{subset}{exp}.json"
                                    if si_ratio:
                                        left, right = fixed_silence
                                        novad_output_manifest= f"{final_output_folder}/{mode}/{lang}/asr_{vad_exp}_{model}_output_manifest_{db}_{subset}{exp}_{left}_{right}.json"
                                    if not si_ratio:
                                        left, right = 0, 0
                                    if use_model_path:
                                        if 'conformer_ctc' in asr_model:
                                            # RIVA: buffered, with a chunk size of 4.8s, buffer size of 8s
                                            # NeMo mimic offline chunk_len_in_ms 22

                                            os.system(f'python ../asr_chunked_inference/ctc/speech_to_text_buffered_infer_ctc.py \
                                                --asr_model {asr_model} \
                                                --test_manifest {input_manifest} \
                                                --chunk_len_in_ms 20000 \
                                                --output_path {novad_output_manifest} \
                                                --batch_size 128 \
                                                --model_stride 4 \
                                                --total_buffer_in_secs 22')
                                        elif 'conformer_transducer' in asr_model:
                                            os.system(f'python ../asr_chunked_inference/rnnt/speech_to_text_buffered_infer_rnnt.py \
                                                --asr_model {asr_model} \
                                                --test_manifest {input_manifest} \
                                                --chunk_len_in_secs 2.2 \
                                                --output_path {novad_output_manifest} \
                                                --batch_size 128 \
                                                --model_stride 4 \
                                                --total_buffer_in_secs 22')
                                        else:
                                            os.system(f'python ../transcribe_speech.py \
                                                model_path={asr_model} \
                                                dataset_manifest={input_manifest} \
                                                batch_size=32 \
                                                amp=True \
                                                output_filename={novad_output_manifest} \
                                                left={left} \
                                                right={right}') 
                                        WER, WER_nospace = evaluate_asr(novad_output_manifest, use_cer=use_cer, no_space=no_space, clean_text=clean_text)
                                        print(f"no vad WER is {WER}, no vad WER no_space is {WER_nospace}")

                                    else:
                                        os.system(f'python ../transcribe_speech.py \
                                            pretrained_name={asr_model} \
                                            dataset_manifest={input_manifest} \
                                            batch_size=32 \
                                            amp=True \
                                            output_filename={novad_output_manifest} \
                                            left={left} \
                                            right={right}') 
                                        WER, WER_nospace = evaluate_asr(novad_output_manifest, use_cer=use_cer, no_space=no_space, clean_text=clean_text)
                                        print(f"no vad WER is {WER}, no vad WER no_space is {WER_nospace}")

                                elif vad_exp in ["neural_vad", "energy_vad", "oracle_vad", "random_vad"]:
                                    if not si_ratio:
                                        left, right = 0, 0

                                    vad_out_manifest_filepath= os.path.join(mode_lang_folder, f"vad_out_{vad_exp}.json")

                                    if vad_exp=="neural_vad":
                                        params = {
                                            "onset": 0.5,
                                            "offset": 0.3,
                                            "min_duration_on": 0.5,
                                            "min_duration_off": 0.5,
                                            "pad_onset": 0.2,
                                            "pad_offset": 0.2,
                                        }
                                        # params = {
                                        #     "onset": 0.7,
                                        #     "offset": 0.4,
                                        #     "min_duration_on": 1,
                                        #     "min_duration_off": 0.5,
                                        #     "pad_onset": 0.2,
                                        #     "pad_offset": 0.2,
                                        # }
                                        # params = {
                                        #     "onset": 0.5,
                                        #     "offset": 0.5,
                                        #     "min_duration_on": 0.5,
                                        #     "min_duration_off": 0.5,
                                        #     "pad_onset": 0.2,
                                        #     "pad_offset": -0.2
                                        # }
                                        # params = {
                                        #     "onset": 0.4,
                                        #     "offset": 0.9,
                                        #     "min_duration_on": 0,
                                        #     "min_duration_off": 1.0,
                                        #     "pad_onset": 0.2,
                                        #     "pad_offset": -0.2
                                        # }
                                        # vad_model="/home/fjia/models/mVAD_lin_nonoise_marblenet-3x2x64-4N-256bs-50e-0.02lr-0.001wd/slurm_mVAD_lin_nonoise_marblenet-3x2x64-4N-256bs-50e-0.02lr-0.001wd/checkpoints/mVAD_lin_nonoise_marblenet-3x2x64-4N-256bs-50e-0.02lr-0.001wd.nemo" # here we use vad_marblenet for example, you can choose other VAD models.
                                        vad_model="/home/fjia/models/mVAD_lin_marblenet-3x2x64-4N-256bs-50e-0.02lr-0.001wd/slurm_mVAD_lin_marblenet-3x2x64-4N-256bs-50e-0.02lr-0.001wd/checkpoints/mVAD_lin_marblenet-3x2x64-4N-256bs-50e-0.02lr-0.001wd.nemo" # here we use vad_marblenet for example, you can choose other VAD models.
                                        
                                        if save_neural_vad:
                                            frame_out_dir = f"{final_output_folder}/{mode}/{lang}/{model}/neural_vad_{db}"
                                            # if os.path.exists(frame_out_dir):
                                            #     print(f"!! deleting existing {frame_out_dir}")
                                            #     shutil.rmtree(frame_out_dir) 
                                        else:
                                            frame_out_dir = os.path.join(mode_lang_folder, "neural_vad")
                                        os.system(f'python vad_infer.py --config-path="../conf/vad" --config-name="vad_inference_postprocessing.yaml" \
                                            num_workers={num_workers} \
                                            dataset={input_manifest} \
                                            vad.model_path={vad_model} \
                                            frame_out_dir={frame_out_dir} \
                                            vad.parameters.window_length_in_sec=0.63 \
                                            vad.parameters.shift_length_in_sec={shift_length_in_sec} \
                                            vad.parameters.smoothing={smoothing} \
                                            vad.parameters.overlap={overlap} \
                                            vad.parameters.postprocessing.onset={params["onset"]} \
                                            vad.parameters.postprocessing.offset={params["offset"]} \
                                            vad.parameters.postprocessing.min_duration_on={params["min_duration_on"]} \
                                            vad.parameters.postprocessing.min_duration_off={params["min_duration_off"]} \
                                            vad.parameters.postprocessing.pad_onset={params["pad_onset"]} \
                                            vad.parameters.postprocessing.pad_offset={params["pad_offset"]} \
                                            out_manifest_filepath={vad_out_manifest_filepath}')  

                                    elif vad_exp=="energy_vad":
                                        vad_out_manifest_filepath = perform_energy_vad(input_manifest, vad_out_manifest_filepath)

                                    else: # random_vad, oracle_vad and energy_oracle_vad
                                        vad_out_manifest_filepath = write_ss2manifest(input_manifest, vad_exp, vad_out_manifest_filepath, random_vad_ref=random_vad_ref)

                                    segmented_output_manifest = os.path.join(mode_lang_folder, "asr_segmented_output_manifest.json")

                                    if use_model_path:
                                        os.system(f'python ../transcribe_speech.py \
                                            model_path={asr_model} \
                                            dataset_manifest={vad_out_manifest_filepath} \
                                            batch_size=32 \
                                            amp=True \
                                            output_filename={segmented_output_manifest} \
                                            left={left} \
                                            right={right}')
                                    else:
                                        os.system(f'python ../transcribe_speech.py \
                                            pretrained_name={asr_model} \
                                            dataset_manifest={vad_out_manifest_filepath} \
                                            batch_size=32 \
                                            amp=True \
                                            output_filename={segmented_output_manifest} \
                                            left={left} \
                                            right={right}')

                                    stitched_output_manifest = os.path.join(mode_lang_folder, "stitched_asr_output_manifest.json")
                                    speech_segments_tensor_dir = os.path.join(mode_lang_folder, f"{vad_exp}/speech_segments_{db}")
                                    stitched_output_manifest = stitch_segmented_asr_output(segmented_output_manifest, speech_segments_tensor_dir, stitched_output_manifest)

                                    aligned_vad_asr_output_manifest = f"{final_output_folder}/{mode}/{lang}/asr_{vad_exp}_{model}_output_manifest_{db}_{subset}{exp}.json"
                                    aligned_vad_asr_output_manifest = construct_manifest_eval(input_manifest, stitched_output_manifest, aligned_vad_asr_output_manifest, use_cer)

                                    # DetER, FA, MISS = evaluate_vad(aligned_vad_asr_output_manifest, ref=ref)
                                    # print(f'DetER (%) : {DetER}, FA (%): {FA}, MISS (%): {MISS}')
                                    DetER, FA, MISS = None, None, None
                                
                                    WER, WER_nospace = evaluate_asr(aligned_vad_asr_output_manifest, use_cer=use_cer, no_space=no_space, clean_text=clean_text)
                                    print(f"vad WER is {WER}, vad WER no_space is {WER_nospace}")

                                    os.remove(vad_out_manifest_filepath)
                                    os.remove(stitched_output_manifest)
                                    os.remove(segmented_output_manifest)
                                    # shutil.rmtree(speech_segments_tensor_dir)

                                else:
                                    raise ValueError(f"vad_exp could only be in novad, energy_vad, neural_vad and oracle_vad but got {vad_exp}")


                            elif mode == "streaming":
                                chunk_len_in_ms = 160 
                                if vad_exp =="novad":
                                    novad_output_manifest= f"{final_output_folder}/{mode}/{lang}/asr_{vad_exp}_{model}_output_manifest_{db}.json"
                                
                                    os.system(f'python ../asr_chunked_inference/ctc/speech_to_text_buffered_infer_ctc.py \
                                    --asr_model {asr_model} \
                                    --test_manifest {input_manifest} \
                                    --chunk_len_in_ms {chunk_len_in_ms} \
                                    --output_path {novad_output_manifest} \
                                    --batch_size 128 \
                                    --model_stride {model_stride} \
                                    --total_buffer_in_secs 4')

                                    WER, WER_nospace = evaluate_asr(novad_output_manifest, use_cer=use_cer, no_space=no_space, clean_text=clean_text)
                                    print(f"no vad WER is {WER}, no vad WER no_space is {WER_nospace}")

                                elif vad_exp in ["neural_vad", "energy_vad", "oracle_vad"]:
                                    vad_out_manifest_filepath= os.path.join(mode_lang_folder, f"vad_out_{vad_exp}.json")
                                    vad_asr_output_manifest= f"{final_output_folder}/{mode}/{lang}/asr_{vad_exp}_{model}_output_manifest_{db}.json"
                                    
                                    aligned_vad_asr_output_manifest = f"{final_output_folder}/{mode}/{lang}/asr_{vad_exp}_{model}_output_manifest_{db}.json"
                                    if si_ratio:
                                        aligned_vad_asr_output_manifest= f"{final_output_folder}/{mode}/{lang}/asr_{vad_exp}_{model}_output_manifest_{db}_{fixed_silence[0]}_{fixed_silence[1]}.json"


                                    if vad_exp=="neural_vad":
                                        # vad_model="/home/fjia/models/mVAD_lin_nonoise_marblenet-3x2x64-4N-256bs-50e-0.02lr-0.001wd/slurm_mVAD_lin_nonoise_marblenet-3x2x64-4N-256bs-50e-0.02lr-0.001wd/checkpoints/mVAD_lin_nonoise_marblenet-3x2x64-4N-256bs-50e-0.02lr-0.001wd.nemo" # here we use vad_marblenet for example, you can choose other VAD models.
                                        vad_model="/home/fjia/models/mVAD_lin_marblenet-3x2x64-4N-256bs-50e-0.02lr-0.001wd/slurm_mVAD_lin_marblenet-3x2x64-4N-256bs-50e-0.02lr-0.001wd/checkpoints/mVAD_lin_marblenet-3x2x64-4N-256bs-50e-0.02lr-0.001wd.nemo" # here we use vad_marblenet for example, you can choose other VAD models.
                                        threshold = 0.5 # same as onset offset
                                        look_back = 4
                                        if save_neural_vad:
                                            frame_out_dir = f"{final_output_folder}/{mode}/{lang}/{model}/neural_vad_{db}"
                                        else:
                                            frame_out_dir = os.path.join(mode_lang_folder, "neural_vad")

                                        os.system(f'python ../asr_chunked_inference/ctc/speech_to_text_buffered_infer_ctc.py \
                                        --asr_model {asr_model} \
                                        --vad_model {vad_model} \
                                        --test_manifest {input_manifest} \
                                        --chunk_len_in_ms {chunk_len_in_ms} \
                                        --output_path {vad_asr_output_manifest} \
                                        --batch_size 128 \
                                        --model_stride {model_stride} \
                                        --total_buffer_in_secs 4 \
                                        --threshold {threshold} \
                                        --look_back {look_back} \
                                        --vad_before_asr')

                                        aligned_vad_asr_output_manifest = construct_manifest_eval(input_manifest, vad_asr_output_manifest, aligned_vad_asr_output_manifest, use_cer)
                                    
                                    elif vad_exp=="energy_vad":
                                        # no look back
                                        vad_out_manifest_filepath = perform_energy_vad(input_manifest, vad_out_manifest_filepath)

                                        os.system(f'python ../asr_chunked_inference/ctc/speech_to_text_buffered_infer_ctc.py \
                                        --asr_model {asr_model} \
                                        --test_manifest {vad_out_manifest_filepath} \
                                        --chunk_len_in_ms {chunk_len_in_ms} \
                                        --output_path {vad_asr_output_manifest} \
                                        --batch_size 128 \
                                        --model_stride {model_stride} \
                                        --total_buffer_in_secs 4')

                                        stitched_output_manifest = os.path.join(mode_lang_folder, "stitched_asr_output_manifest.json")
                                        stitched_output_manifest = stitch_segmented_asr_output(
                                            vad_asr_output_manifest,
                                            speech_segments_tensor_dir = os.path.join(mode_lang_folder, "speech_segments"),
                                            stitched_output_manifest = stitched_output_manifest)
                                        aligned_vad_asr_output_manifest = construct_manifest_eval(input_manifest, stitched_output_manifest, aligned_vad_asr_output_manifest, use_cer)
                                        

                                    else: # oracle_vad and energy_oracle_vad
                                        vad_out_manifest_filepath = write_ss2manifest(input_manifest, vad_exp, vad_out_manifest_filepath)

                                        os.system(f'python ../asr_chunked_inference/ctc/speech_to_text_buffered_infer_ctc.py \
                                        --asr_model {asr_model} \
                                        --test_manifest {vad_out_manifest_filepath} \
                                        --chunk_len_in_ms {chunk_len_in_ms} \
                                        --output_path {vad_asr_output_manifest} \
                                        --batch_size 128 \
                                        --model_stride {model_stride} \
                                        --total_buffer_in_secs 4')

                                        stitched_output_manifest = os.path.join(mode_lang_folder, "stitched_asr_output_manifest.json")
                                        stitched_output_manifest = stitch_segmented_asr_output(
                                            vad_asr_output_manifest,
                                            speech_segments_tensor_dir = os.path.join(mode_lang_folder, "speech_segments"),
                                            stitched_output_manifest = stitched_output_manifest)
                                        aligned_vad_asr_output_manifest = construct_manifest_eval(input_manifest, stitched_output_manifest, aligned_vad_asr_output_manifest, use_cer)
                                    
                                    
                                    # DetER, FA, MISS = evaluate_vad(aligned_vad_asr_output_manifest, ref=ref)
                                    # print(f'DetER (%) : {DetER}, FA (%): {FA}, MISS (%): {MISS}')
                                    DetER, FA, MISS = None, None, None

                                    WER, WER_nospace = evaluate_asr(aligned_vad_asr_output_manifest, use_cer=use_cer, no_space=no_space, clean_text=clean_text)
                                    print(f"vad WER is {WER}, vad WER no_space is {WER_nospace}")

                                else:
                                    raise ValueError(f"vad_exp could only be in novad, energy_vad, neural_vad and oracle_vad but got {vad_exp}")

                            else:
                                raise ValueError(f"Invalid mode {mode}. Mode could be either streaming or offline.")

                            end = time.time()
                            run_time = end-start
                            # collecting evaluation result
                            with open(res_file, "a") as fp:
                                if si_ratio:
                                    if vad_exp == "novad" or vad_exp=="oracle_vad":
                                        fp.write(f"{subset},{mode},{lang},{model},{db},{vad_exp},{round(WER, 4)},{round(WER_nospace, 4)},{round(run_time, 4)},{left},{right}")
                                        fp.write("\n")
                                else:
                                    if vad_exp == "novad" or vad_exp=="oracle_vad" or vad_exp=="random_vad":
                                        fp.write(f"{subset},{mode},{lang},{model},{db},{vad_exp},{round(WER, 4)},{round(WER_nospace, 4)},{round(run_time, 4)}")
                                        fp.write("\n")
                                    elif vad_exp == "energy_vad" :
                                        fp.write(f'{subset},{mode},{lang},{model},{db},{vad_exp},{round(WER, 4)},{round(WER_nospace, 4)},{round(run_time, 4)},{DetER},{FA},{MISS}')
                                        fp.write("\n")
                                    else:
                                        if mode == 'streaming':
                                            # think about how to convert patience to min_duration_on off and look back to pad
                                            fp.write(f'{subset},{mode},{lang},{model},{db},{vad_exp},{round(WER, 4)},{round(WER_nospace, 4)},{round(run_time, 4)},{DetER},{FA},{MISS}')
                                        else:
                                            fp.write(f'{subset},{mode},{lang},{model},{db},{vad_exp},{round(WER, 4)},{round(WER_nospace, 4)},{round(run_time, 4)},{DetER},{FA},{MISS},{round(params["onset"], 4)},{round(params["offset"], 4)},{round(params["min_duration_on"], 4)},{round(params["min_duration_off"], 4)},{round(params["pad_onset"], 4)},{round(params["pad_offset"], 4)}')
                                        fp.write("\n")


                
if __name__ == '__main__':
    main()