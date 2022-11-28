import json
import multiprocessing
import os
import shutil
import time
from typing import Tuple

import numpy as np
import torch
import tqdm
from extract_speech import process_one_file
from pyannote.core import Annotation, Segment
from pyannote.metrics import detection

from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.collections.asr.parts.utils.vad_utils import (
    construct_manifest_eval,
    plot,
    stitch_segmented_asr_output,
    write_ss2manifest,
)


def evaluate_vad(manifest: str, ref: str = "oracle_vad") -> Tuple[float, float, float]:
    metric = detection.DetectionErrorRate()
    
    for line in open(manifest, 'r', encoding='utf-8'):
        sample = json.loads(line)

        reference = Annotation()
        vad_ref = torch.load(sample[ref])
        for row in vad_ref:
            reference[Segment(row[0], row[1])] = 'Speech'
        
        hypothesis = Annotation()
        if sample['speech_segments_filepath'] != "":
            if sample['speech_segments_filepath'].endswith(".pt"):
                vad_hyp = torch.load(sample['speech_segments_filepath'])
            else:
                vad_hyp = np.load(sample['speech_segments_filepath'])
            for row in vad_hyp:
                hypothesis[Segment(row[0], row[1])] = 'Speech'

        metric(reference, hypothesis) 

    # vad evaluation
    report = metric.report(display=False)
    DetER = report.iloc[[-1]][('detection error rate', '%')].item()
    FA = report.iloc[[-1]][('false alarm', '%')].item()
    MISS = report.iloc[[-1]][('miss', '%')].item()
    return DetER, FA, MISS


def perform_energy_vad(input_manifest, output_manifest="generated_energy_ss_manifest.json"):
    data = []
    for line in open(input_manifest, 'r', encoding='utf-8'):
        data.append(json.loads(line))
       
    number_of_processes = 15
    p = multiprocessing.Pool(number_of_processes)
    results = []
    for result in tqdm.tqdm(p.imap_unordered(process_one_file, data), total=len(data)):
        results.append(result)
        
    p.close()
    
    with open(output_manifest, "w") as fout:
        for result in results:
            # each file might have multi meta
            for meta in result:
                json.dump(meta, fout)
                fout.write('\n')
                fout.flush()
            
    return output_manifest



def switch_lang_model(lang: str, model: str) -> Tuple[bool, bool, bool, str]:
    lang_model_table = {
        'english-riva_conformer_ctc':  (False, False, True, "/home/fjia/code/models/riva/en-US/Conformer-CTC-L-en-US-ASR-set-4p0.nemo"), 
        'english-citrinet_2.0':  (False, False, True, ""),
        'english-citrinet_ngc':  (False, False, False, "stt_en_citrinet_1024_gamma_0_25"),
        'english-nr_citrinet':  (False, False, True, "/home/fjia/models/english/Citrinet_Aug_1024_Gamma_0-25_NeMo_ASRSET_2.0_e200.nemo"),
        'english-riva_citrinet':  (False, False, True, "/home/fjia/code/models/CitriNet-1024-8x-Stride-Gamma-0.25.nemo"),
        'english-riva_citrinet_new':  (False, False, True, "/home/fjia/code/models/riva/en-US/Citrinet-1024-SPE-Unigram-1024-Jarvis-ASRSet-3.0_no_weight_decay_e100-averaged.nemo"),
        'english-conformer_transducer':  (False, False, False, "stt_en_conformer_transducer_large"),
        'english-nr_conformer_transducer':  (False, False, True, "/home/fjia/models/jagadeesh_nr/english/aug/rno8_bucket32_Aug_nemo2.0_d512_adamwlr5.0_wd1e-3_aug10x0.05spunigram1024_emit0_nodes32_bucketing_200e_b4.nemo"),
        'mandarin-citrinet': (True, True, False, "stt_zh_citrinet_1024_gamma_0_25"), # test 5000
        'french-citrinet':   (False, False, False, "stt_fr_citrinet_1024_gamma_0_25"), # test 2320
        'german-citrinet':   (False, False, False, "stt_de_citrinet_1024"), #dev 15845 test 15854
        'spanish-nr_citrinet':  (False, False, True, "/home/fjia/models/jagadeesh_nr/spanish/finetuning_with_augmentation/stt_es_citrinet_1024_gamma_0_25.nemo"), 
        'spanish-nr_conformer_ctc':  (False, False, True, "/home/fjia/models/jagadeesh_nr/spanish/finetuning_with_augmentation/stt_es_conformer_ctc_large.nemo"), 
        'spanish-nr_conformer_transducer':  (False, False, True, "/home/fjia/models/jagadeesh_nr/spanish/finetuning_with_augmentation/stt_es_conformer_transducer_large.nemo"), 
        'spanish-nr_contextnet':  (False, False, True, "/home/fjia/models/jagadeesh_nr/spanish/finetuning_with_augmentation/stt_es_contextnet_1024.nemo"), 
        'russian-citrinet':  (False, False, True, "/home/fjia/models/vitaly/ru_models/CitriNet-1024-8x-Stride-Gamma-0.25.nemo"), #dev 9361 test 9355 TODO will ask vitaly about new checkpoint  
    }
    


    lang_model = lang + "-" + model
    return lang_model_table.get(lang_model, None)


def switch_model_buffered_param(model: str) -> float:
     # TODO other models
    model_streaming_param_table = {
        'citrinet': 8 ,
        'conformer_ctc': 4
    }
    if "nr" in model:
        model = model.split("nr_")[1]
    if "riva" in model:
        # model = model.split("riva_")[1]
        model = model.split("riva_")[1].split("_")[0]

    return model_streaming_param_table.get(model, None)


def evaluate_asr(manifest: str, use_cer: bool=False, no_space:bool =False, clean_text=False) -> Tuple[float,float]:
    predicted_text, ground_truth_text = [], []
    predicted_text_nospace, ground_truth_text_nospace = [], []

    for line in open(manifest, 'r', encoding='utf-8'):
        sample = json.loads(line)
        predicted_text.append(sample['pred_text'])
        predicted_text_nospace.append(sample['pred_text'].replace(" ", ""))

        text = sample['text'] 
        if clean_text:
            text = clean_label(text)

        ground_truth_text.append(text)
        ground_truth_text_nospace.append(text.replace(" ", ""))
    # asr evaluation
    WER_nospace = 1.0
    # if ground_truth_text == "" and predicted_text == "":
    #     print("both empty")
    #     WER = 0
    # else:
    WER = word_error_rate(hypotheses=predicted_text, references=ground_truth_text, use_cer=use_cer)
    #todo
    if no_space:
        WER_nospace = word_error_rate(hypotheses=predicted_text_nospace, references=ground_truth_text_nospace, use_cer=use_cer)
    return WER, WER_nospace


def clean_label(_str, num_to_words=True):
    """
    Remove unauthorized characters in a string, lower it and remove unneeded spaces
    Parameters
    ----------
    _str : the original string
    Returns
    -------
    string
    """
    replace_with_space = [char for char in '/?*\",.:=?_{|}~¨«·»¡¿„…‧‹›≪≫!:;ː→']
    replace_with_blank = [char for char in '`¨´‘’“”`ʻ‘’“"‘”']
    replace_with_apos = [char for char in '‘’ʻ‘’‘']
    _str = _str.strip()
    _str = _str.lower()
    for i in replace_with_blank:
        _str = _str.replace(i, "")
    for i in replace_with_space:
        _str = _str.replace(i, " ")
    for i in replace_with_apos:
        _str = _str.replace(i, "'")
    if num_to_words:
        _str = convert_num_to_words(_str)
    return " ".join(_str.split())

   
def convert_num_to_words(_str):
    """
    Convert digits to corresponding words
    Parameters
    ----------
    _str : the original string
    Returns
    -------
    string
    """
    num_to_words = ["zero", "one", "two", "three", "four", "five",
                    "six", "seven", "eight", "nine"]
    _str = _str.strip()
    words = _str.split()
    out_str = ""
    num_word = []
    for word in words:
        if word.isnumeric():
            num = int(word)
            while(num):
                digit = num % 10
                digit_word = num_to_words[digit]
                num_word.append(digit_word)
                num = int(num / 10)
                if not(num):
                    num_str = ""
                    num_word = num_word[::-1]
                    for ele in num_word:
                        num_str += ele + " "
                    out_str += num_str + " "
                    num_word.clear()
        else:
            out_str += word + " "
    out_str = out_str.strip()
    return out_str
