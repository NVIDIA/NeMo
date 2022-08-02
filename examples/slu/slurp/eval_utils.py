import ast
import contextlib
import glob
import json
import os
from dataclasses import dataclass, is_dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import pytorch_lightning as pl
import torch
from evaluation.metrics import ErrorMetric
from omegaconf import MISSING, OmegaConf
from tqdm.auto import tqdm

from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.parts.utils.slu_utils import SearcherConfig
from nemo.core.config import hydra_runner
from nemo.utils import logging, model_utils


def parse_entity(item: Dict):
    error = False
    for key in ["type", "filler"]:
        if key not in item or not isinstance(item[key], str):
            item[key] = "none"
            error = True
    return item, error


def parse_semantics_str2dict(semantics_str: Union[List[str], str]) -> Dict:
    invalid = False
    if isinstance(semantics_str, dict):
        return semantics_str, invalid
    if isinstance(semantics_str, list):
        semantics_str = " ".join(semantics_str)

    try:
        _dict = ast.literal_eval(semantics_str.replace("|", ","))
        if not isinstance(_dict, dict):
            _dict = {
                "scenario": "none",
                "action": "none",
                "entities": [],
            }
            invalid = True
    except:  # need this if the output is not a valid dictionary
        _dict = {
            "scenario": "none",
            "action": "none",
            "entities": [],
        }
        invalid = True

    if "scenario" not in _dict or not isinstance(_dict["scenario"], str):
        _dict["scenario"] = "none"
        invalid = True
    if "action" not in _dict or not isinstance(_dict["action"], str):
        _dict["action"] = "none"
        invalid = True
    if "entities" not in _dict:
        _dict["entities"] = []
        invalid = True
    else:
        for i, x in enumerate(_dict["entities"]):
            item, entity_error = parse_entity(x)
            invalid = invalid or entity_error
            _dict["entities"][i] = item

    return _dict, invalid


class SLURPEvaluator:
    def __init__(self, average_mode: str = 'micro') -> None:
        if average_mode not in ['micro', 'macro']:
            raise ValueError(f"Only supports 'micro' or 'macro' average, but got {average_mode} instead.")
        self.average_mode = average_mode
        self.scenario_f1 = ErrorMetric.get_instance(metric="f1", average=average_mode)
        self.action_f1 = ErrorMetric.get_instance(metric="f1", average=average_mode)
        self.intent_f1 = ErrorMetric.get_instance(metric="f1", average=average_mode)
        self.span_f1 = ErrorMetric.get_instance(metric="span_f1", average=average_mode)
        self.distance_metrics = {}
        for distance in ['word', 'char']:
            self.distance_metrics[distance] = ErrorMetric.get_instance(
                metric="span_distance_f1", average=average_mode, distance=distance
            )
        self.slu_f1 = ErrorMetric.get_instance(metric="slu_f1", average=average_mode)
        self.invalid = 0
        self.total = 0

    def reset(self):
        self.scenario_f1 = ErrorMetric.get_instance(metric="f1", average=self.average_mode)
        self.action_f1 = ErrorMetric.get_instance(metric="f1", average=self.average_mode)
        self.intent_f1 = ErrorMetric.get_instance(metric="f1", average=self.average_mode)
        self.span_f1 = ErrorMetric.get_instance(metric="span_f1", average=self.average_mode)
        self.distance_metrics = {}
        for distance in ['word', 'char']:
            self.distance_metrics[distance] = ErrorMetric.get_instance(
                metric="span_distance_f1", average=self.average_mode, distance=distance
            )
        self.slu_f1 = ErrorMetric.get_instance(metric="slu_f1", average=self.average_mode)
        self.invalid = 0
        self.total = 0

    def update(self, predictions: Union[List[str], str], groundtruth: Union[List[str], str]) -> None:
        if isinstance(predictions, str):
            predictions = [predictions]
        if isinstance(groundtruth, str):
            groundtruth = [groundtruth]

        for pred, truth in zip(predictions, groundtruth):
            pred, syntax_error = parse_semantics_str2dict(pred)
            truth, _ = parse_semantics_str2dict(truth)
            self.scenario_f1(truth["scenario"], pred["scenario"])
            self.action_f1(truth["action"], pred["action"])
            self.intent_f1(f"{truth['scenario']}_{truth['action']}", f"{pred['scenario']}_{pred['action']}")
            self.span_f1(truth["entities"], pred["entities"])
            for distance, metric in self.distance_metrics.items():
                metric(truth["entities"], pred["entities"])

            self.total += 1
            self.invalid += int(syntax_error)

    def compute(self, aggregate=True) -> Dict:
        scenario_results = self.scenario_f1.get_metric()
        action_results = self.action_f1.get_metric()
        intent_results = self.intent_f1.get_metric()
        entity_results = self.span_f1.get_metric()
        word_dist_results = self.distance_metrics['word'].get_metric()
        char_dist_results = self.distance_metrics['char'].get_metric()
        self.slu_f1(word_dist_results)
        self.slu_f1(char_dist_results)
        slurp_results = self.slu_f1.get_metric()

        if not aggregate:
            return {
                "scenario": scenario_results,
                "action": action_results,
                "intent": intent_results,
                "entity": entity_results,
                "word_dist": word_dist_results,
                "char_dist": char_dist_results,
                "slurp": slurp_results,
                "invalid": self.invalid,
                "total": self.total,
            }

        scores = dict()
        scores["invalid"] = self.invalid
        scores["total"] = self.total
        self.update_scores_dict(scenario_results, scores, "scenario")
        self.update_scores_dict(action_results, scores, "action")
        self.update_scores_dict(intent_results, scores, "intent")
        self.update_scores_dict(entity_results, scores, "entity")
        self.update_scores_dict(word_dist_results, scores, "word_dist")
        self.update_scores_dict(char_dist_results, scores, "char_dist")
        self.update_scores_dict(slurp_results, scores, "slurp")

        return scores

    def update_scores_dict(self, source: Dict, target: Dict, tag: str = '') -> Dict:
        scores = source['overall']
        p, r, f1 = scores[:3]
        target[f"{tag}_p"] = p
        target[f"{tag}_r"] = r
        target[f"{tag}_f1"] = f1
        return target


@dataclass
class InferenceConfig:
    # Required configs
    model_path: Optional[str] = None  # Path to a .nemo file
    pretrained_name: Optional[str] = None  # Name of a pretrained model
    audio_dir: Optional[str] = None  # Path to a directory which contains audio files
    dataset_manifest: Optional[str] = None  # Path to dataset's JSON manifest

    # General configs
    output_filename: Optional[str] = None
    batch_size: int = 32
    num_workers: int = 8

    # Set `cuda` to int to define CUDA device. If 'None', will look for CUDA
    # device anyway, and do inference on CPU only if CUDA device is not found.
    # If `cuda` is a negative number, inference will be on CPU only.
    cuda: Optional[int] = None
    amp: bool = False
    audio_type: str = "wav"

    # Recompute model transcription, even if the output folder exists with scores.
    overwrite_transcripts: bool = True

    # Decoding strategy for RNNT models
    searcher: SearcherConfig = SearcherConfig(type="greedy")


def slurp_inference(asr_model, path2manifest: str, batch_size: int = 4, num_workers: int = 0,) -> List[str]:

    if num_workers is None:
        num_workers = min(batch_size, os.cpu_count() - 1)

    # We will store transcriptions here
    hypotheses = []
    # Model's mode and device
    mode = asr_model.training
    device = next(asr_model.parameters()).device
    dither_value = asr_model.preprocessor.featurizer.dither
    pad_to_value = asr_model.preprocessor.featurizer.pad_to

    try:
        asr_model.preprocessor.featurizer.dither = 0.0
        asr_model.preprocessor.featurizer.pad_to = 0
        # Switch model to evaluation mode
        asr_model.eval()

        logging_level = logging.get_verbosity()
        logging.set_verbosity(logging.WARNING)

        config = {
            'manifest_filepath': path2manifest,
            'batch_size': batch_size,
            'num_workers': num_workers,
        }

        temporary_datalayer = asr_model._setup_transcribe_dataloader(config)
        for test_batch in tqdm(temporary_datalayer, desc="Transcribing"):
            predictions = asr_model.predict(
                input_signal=test_batch[0].to(device), input_signal_length=test_batch[1].to(device)
            )

            hypotheses += predictions

            del predictions
            del test_batch

    finally:
        # set mode back to its original value
        asr_model.train(mode=mode)
        asr_model.preprocessor.featurizer.dither = dither_value
        asr_model.preprocessor.featurizer.pad_to = pad_to_value
        logging.set_verbosity(logging_level)
    return hypotheses


@hydra_runner(config_name="InferenceConfig", schema=InferenceConfig)
def run_inference(cfg: InferenceConfig) -> InferenceConfig:
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    if is_dataclass(cfg):
        cfg = OmegaConf.structured(cfg)

    if cfg.model_path is None and cfg.pretrained_name is None:
        raise ValueError("Both cfg.model_path and cfg.pretrained_name cannot be None!")
    if cfg.audio_dir is None and cfg.dataset_manifest is None:
        raise ValueError("Both cfg.audio_dir and cfg.dataset_manifest cannot be None!")

    # setup GPU
    if cfg.cuda is None:
        if torch.cuda.is_available():
            device = [0]  # use 0th CUDA device
            accelerator = 'gpu'
        else:
            device = 1
            accelerator = 'cpu'
    else:
        device = [cfg.cuda]
        accelerator = 'gpu'

    map_location = torch.device('cuda:{}'.format(device[0]) if accelerator == 'gpu' else 'cpu')

    # setup model
    if cfg.model_path is not None:
        # restore model from .nemo file path
        model_cfg = ASRModel.restore_from(restore_path=cfg.model_path, return_config=True)
        classpath = model_cfg.target  # original class path
        imported_class = model_utils.import_class_by_path(classpath)  # type: ASRModel
        logging.info(f"Restoring model : {imported_class.__name__}")
        asr_model = imported_class.restore_from(
            restore_path=cfg.model_path, map_location=map_location
        )  # type: ASRModel
        model_name = os.path.splitext(os.path.basename(cfg.model_path))[0]
    else:
        # restore model by name
        asr_model = ASRModel.from_pretrained(
            model_name=cfg.pretrained_name, map_location=map_location
        )  # type: ASRModel
        model_name = cfg.pretrained_name

    trainer = pl.Trainer(devices=device, accelerator=accelerator)
    asr_model.set_trainer(trainer)
    asr_model = asr_model.eval()

    # Setup decoding strategy
    if hasattr(asr_model, 'set_decoding_strategy'):
        asr_model.set_decoding_strategy(cfg.searcher)

    # get audio filenames
    if cfg.audio_dir is not None:
        filepaths = list(glob.glob(os.path.join(cfg.audio_dir, f"**/*.{cfg.audio_type}"), recursive=True))
    else:
        # get filenames from manifest
        filepaths = []
        if os.stat(cfg.dataset_manifest).st_size == 0:
            logging.error(f"The input dataset_manifest {cfg.dataset_manifest} is empty. Exiting!")
            return None

        manifest_dir = Path(cfg.dataset_manifest).parent
        with open(cfg.dataset_manifest, 'r') as f:
            has_two_fields = []
            for line in f:
                item = json.loads(line)
                if "offset" in item and "duration" in item:
                    has_two_fields.append(True)
                else:
                    has_two_fields.append(False)
                audio_file = Path(item['audio_filepath'])
                if not audio_file.is_file() and not audio_file.is_absolute():
                    audio_file = manifest_dir / audio_file
                filepaths.append(str(audio_file.absolute()))

    logging.info(f"\nStart inference with {len(filepaths)} files...\n")

    # setup AMP (optional)
    if cfg.amp and torch.cuda.is_available() and hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
        logging.info("AMP enabled!\n")
        autocast = torch.cuda.amp.autocast
    else:

        @contextlib.contextmanager
        def autocast():
            yield

    # Compute output filename
    if cfg.output_filename is None:
        # create default output filename
        if cfg.audio_dir is not None:
            cfg.output_filename = os.path.dirname(os.path.join(cfg.audio_dir, '.')) + '.json'
        else:
            cfg.output_filename = cfg.dataset_manifest.replace('.json', f'_{model_name}.json')

    # if transcripts should not be overwritten, and already exists, skip re-transcription step and return
    if not cfg.overwrite_transcripts and os.path.exists(cfg.output_filename):
        logging.info(
            f"Previous transcripts found at {cfg.output_filename}, and flag `overwrite_transcripts`"
            f"is {cfg.overwrite_transcripts}. Returning without re-transcribing text."
        )

        return cfg

    # transcribe audio
    with autocast():
        with torch.no_grad():
            predictions = slurp_inference(
                asr_model=asr_model,
                path2manifest=cfg.dataset_manifest,
                batch_size=cfg.batch_size,
                num_workers=cfg.num_workers,
            )

    logging.info(f"Finished transcribing {len(filepaths)} files !")

    logging.info(f"Writing transcriptions into file: {cfg.output_filename}")

    # write audio transcriptions
    with open(cfg.output_filename, 'w', encoding='utf-8') as f:
        if cfg.audio_dir is not None:
            for idx, text in enumerate(predictions):
                item = {'audio_filepath': filepaths[idx], 'pred_text': text}
                f.write(json.dumps(item) + "\n")
        else:
            with open(cfg.dataset_manifest, 'r') as fr:
                for idx, line in enumerate(fr):
                    item = json.loads(line)
                    item['pred_text'] = predictions[idx]
                    f.write(json.dumps(item) + "\n")

    logging.info("Finished writing predictions !")
    return cfg


if __name__ == '__main__':
    run_inference()  # noqa pylint: disable=no-value-for-parameter
