import json
import subprocess
from pathlib import Path
from typing import Tuple

from omegaconf import DictConfig, OmegaConf, open_dict

from nemo.collections.asr.metrics.wer import word_error_rate_detail
from nemo.utils import logging


def run_asr_inference(cfg: DictConfig) -> DictConfig:
    if (cfg.model_path and cfg.pretrained_name) or (not cfg.model_path and not cfg.pretrained_name):
        raise ValueError("Please specify either cfg.model_path or cfg.pretrained_name!")

    if cfg.inference_mode.mode == "offline":
        # Specify default total_buffer_in_secs=22 and chunk_len_in_secs=20 for offline conformer
        # to avoid problem of long audio sample.
        OmegaConf.set_struct(cfg, True)
        if (cfg.model_path and 'conformer' in cfg.model_path.lower()) or (
            cfg.pretrained_name and 'conformer' in cfg.pretrained_name.lower()
        ):
            if 'total_buffer_in_secs' not in cfg.inference_mode or not cfg.inference_mode.total_buffer_in_secs:
                with open_dict(cfg):
                    cfg.inference_mode.total_buffer_in_secs = 22
            if 'chunk_len_in_secs' not in cfg.inference_mode or not cfg.inference_mode.chunk_len_in_secs:
                with open_dict(cfg):
                    cfg.inference_mode.chunk_len_in_secs = 20

            cfg = run_chunked_inference(cfg)
        else:
            cfg = run_offline_inference(cfg)

    elif cfg.inference_mode.mode == "chunked":
        if (
            "total_buffer_in_secs" not in cfg.inference_mode
            or "chunk_len_in_secs" not in cfg.inference_mode
            or not cfg.inference_mode.total_buffer_in_secs
            or not cfg.inference_mode.chunk_len_in_secs
        ):
            raise ValueError(
                f"Please specify both total_buffer_in_secs and chunk_len_in_secs for chunked inference_mode"
            )
        cfg = run_chunked_inference(cfg)

    else:
        raise ValueError(f"inference_mode could only be offline or chunked, but got {cfg.inference_mode.mode}")

    return cfg


def run_chunked_inference(cfg: DictConfig) -> DictConfig:
    if "output_filename" not in cfg or not cfg.output_filename:
        if cfg.model_path:
            model_name = Path(cfg.model_path).setup_model
        else:
            model_name = cfg.pretrained_name
        dataset_name = Path(cfg.test_ds.manifest_filepath).stem
        mode_name = (
            cfg.inference_mode.mode
            + "B"
            + str(cfg.inference_mode.total_buffer_in_secs)
            + "C"
            + str(cfg.inference_mode.chunk_len_in_secs)
        )

        OmegaConf.set_struct(cfg, True)
        with open_dict(cfg):
            cfg.output_filename = model_name + "-" + dataset_name + "-" + mode_name + ".json"

    script_path = (
        Path(__file__).parents[5]
        / "examples"
        / "asr"
        / "asr_chunked_inference"
        / "ctc"
        / "speech_to_text_buffered_infer_ctc.py"
    )

    if (cfg.pretrained_name and 'transducer' in cfg.pretrained_name) or (
        cfg.model_path and 'transducer' in cfg.model_path
    ):
        script_path = (
            Path(__file__).parents[5]
            / "examples"
            / "asr"
            / "asr_chunked_inference"
            / "rnnt"
            / "speech_to_text_buffered_infer_rnnt.py"
        )

    subprocess.run(
        f"python {script_path} "
        f"model_path={cfg.model_path} "
        f"pretrained_name={cfg.pretrained_name} "
        f"dataset_manifest={cfg.test_ds.manifest_filepath} "
        f"output_filename={cfg.output_filename} "
        f"batch_size={cfg.test_ds.batch_size} "
        f"chunk_len_in_secs={cfg.inference_mode.chunk_len_in_secs} "
        f"total_buffer_in_secs={cfg.inference_mode.total_buffer_in_secs} "
        f"model_stride={cfg.inference_mode.model_stride} ",
        shell=True,
        check=True,
    )
    return cfg


def run_offline_inference(cfg: DictConfig) -> DictConfig:
    if "output_filename" not in cfg or not cfg.output_filename:
        if cfg.model_path:
            model_name = Path(cfg.model_path).setup_model
        else:
            model_name = cfg.pretrained_name
        dataset_name = Path(cfg.test_ds.manifest_filepath).stem
        mode_name = cfg.inference_mode.mode

        OmegaConf.set_struct(cfg, True)
        with open_dict(cfg):
            cfg.output_filename = model_name + "-" + dataset_name + "-" + mode_name + ".json"

    script_path = Path(__file__).parents[5] / "examples" / "asr" / "transcribe_speech.py"

    subprocess.run(
        f"python {script_path} "
        f"model_path={cfg.model_path} "
        f"pretrained_name={cfg.pretrained_name} "
        f"dataset_manifest={cfg.test_ds.manifest_filepath} "
        f"output_filename={cfg.output_filename} "
        f"batch_size={cfg.test_ds.batch_size} ",
        shell=True,
        check=True,
    )

    return cfg


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
    num_to_words = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    _str = _str.strip()
    words = _str.split()
    out_str = ""
    num_word = []
    for word in words:
        if word.isnumeric():
            num = int(word)
            while num:
                digit = num % 10
                digit_word = num_to_words[digit]
                num_word.append(digit_word)
                num = int(num / 10)
                if not (num):
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


def cal_write_wer(cfg: DictConfig, pred_text_attr_name: str = None) -> Tuple[DictConfig, dict ]:

    samples = []
    hyps = []
    refs = []

    for line in open(cfg.asr_eval.output_filename, 'r'):
        sample = json.loads(line)

        if 'text' not in sample:
            raise ValueError(
                "ground-truth text does not present in manifest! Cannot calculate Word Error Rate. Exiting!"
            )

        if not pred_text_attr_name:
            pred_text_attr_name = "pred_text"

        hyp = sample[pred_text_attr_name]
        ref = sample['text']

        if cfg.analyst.metric_calculator.clean_groundtruth_text:
            ref = clean_label(ref)

        wer, words, ins_rate, del_rate, sub_rate = word_error_rate_detail(hypotheses=[hyp], references=[ref])
        sample['wer'] = wer
        sample['words'] = words
        sample['ins_rate'] = ins_rate
        sample['del_rate'] = del_rate
        sample['sub_rate'] = sub_rate

        samples.append(sample)
        hyps.append(hyp)
        refs.append(ref)

    # wer = word_error_rate(hypotheses=hyps, references=refs)
    total_wer, total_ref_words, total_ins_rate, total_del_rate, total_sub_rate = word_error_rate_detail(hypotheses=hyps, references=refs)

    if "output_filename" not in cfg.analyst.metric_calculator or not cfg.analyst.metric_calculator.output_filename:
        # overwrite the current manifest
        OmegaConf.set_struct(cfg, True)
        with open_dict(cfg):
            cfg.analyst.metric_calculator.output_filename = cfg.asr_eval.output_filename

    with open(cfg.analyst.metric_calculator.output_filename, 'w') as fout:
        for sample in samples:
            json.dump(sample, fout)
            fout.write('\n')
            fout.flush()

    total_res = {
        "wer": wer,
        "num_ref_words": total_ref_words,
        "ins_rate": total_ins_rate,
        "del_rate": total_del_rate,
        "sub_rate": total_sub_rate
    }
    return  cfg, total_res


def target_metadata_wer(manifest: str, target: str, meta_cfg: DictConfig, eval_metric: str = "wer",) -> dict:

    wer_each_class = {}
    for line in open(manifest, 'r'):
        sample = json.loads(line)
        if target in sample:
            target_class = sample[target]
            if target_class not in wer_each_class:
                wer_each_class[target_class] = {'samples': 0, 'num_ref_words': 0, "errors": 0, "inss": 0, "dels": 0, "subs":0}

            wer_each_class[target_class]['samples'] += 1

            words = sample["num_ref_words"]
            wer_each_class[target_class]["num_ref_words"] += words
            wer_each_class[target_class]["errors"] += words * sample[eval_metric] 
            wer_each_class[target_class]["inss"] += words * sample["ins_rate"] 
            wer_each_class[target_class]["dels"] += words * sample["del_rate"] 
            wer_each_class[target_class]["subs"] += words * sample["sub_rate"] 
            

    if len(wer_each_class) > 0:
        occ_avg_wer = {}
        for target_class in wer_each_class:

            occ_avg_wer[target_class] = {}
            occ_avg_wer[target_class]["samples"] = occ
            occ_avg_wer[target_class]["wer"] = avg_wer

            all_occ += occ
            all_wer += sum(wer_each_class[target_class])

    else:
        logging.info(f"metadata '{target}' does not present in manifest. Skipping! ")
        occ_avg_wer = {'all_class': {'occ': 0, 'avg_wer': None}}



    slot_occ_avg_wer = {}
    if 'slot' in meta_cfg and meta_cfg.slot:
        for target_class in occ_avg_wer:
            if target_class != "all_class":
                for s in meta_cfg.slot:
                    if isinstance(s[0], float) or isinstance(s[0], int):
                        if s[0] <= target_class < s[1]:
                            slot_key = "slot-" + ",".join(str(i) for i in s)
                            if slot_key not in slot_occ_avg_wer:
                                slot_occ_avg_wer[slot_key] = {'occ': 0, 'sum_wer': 0}
                            slot_occ_avg_wer[slot_key]['occ'] += occ_avg_wer[target_class]['occ']
                            slot_occ_avg_wer[slot_key]['sum_wer'] += (
                                occ_avg_wer[target_class]['avg_wer'] * occ_avg_wer[target_class]['occ']
                            )
                            break

                    elif isinstance(s[0], str):
                        if target_class in s:
                            slot_key = "slot-" + ",".join(s)
                            if slot_key not in slot_occ_avg_wer:
                                slot_occ_avg_wer[slot_key] = {'occ': 0, 'sum_wer': 0}
                            slot_occ_avg_wer[slot_key]['occ'] += occ_avg_wer[target_class]['occ']
                            slot_occ_avg_wer[slot_key]['sum_wer'] += (
                                occ_avg_wer[target_class]['avg_wer'] * occ_avg_wer[target_class]['occ']
                            )
                            break

                    else:
                        raise ValueError("Current only support target metadata belongs to numeric or string ")

        for slot_key in slot_occ_avg_wer:
            slot_occ_avg_wer[slot_key]['avg_wer'] = (
                slot_occ_avg_wer[slot_key]['sum_wer'] / slot_occ_avg_wer[slot_key]['occ']
            )
            slot_occ_avg_wer[slot_key].pop('sum_wer')

        occ_avg_wer.update(slot_occ_avg_wer)

    if meta_cfg.wer_each_class:
        return occ_avg_wer

    elif (not meta_cfg.wer_each_class) and ('slot' in meta_cfg and meta_cfg.slot):
        ret = {'all_class': occ_avg_wer['all_class']}
        for i in occ_avg_wer:
            if isinstance(i, str) and i.startswith('slot-'):
                ret[i] = occ_avg_wer[i]
        return ret
    else:
        return {'all_class': occ_avg_wer['all_class']}
