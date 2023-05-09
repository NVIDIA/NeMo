# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
#
# Run ``python build_ensemble.py --help`` for usage examples.


import os
import tempfile
from dataclasses import dataclass, is_dataclass
from typing import List, Optional

# TODO: need fix for import
import joblib
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC
from transcribe_speech import TranscriptionConfig, transcribe_speech

from nemo.collections.asr.models.confidence_ensemble import ConfidenceEnsembleModel
from nemo.collections.asr.parts.utils.asr_confidence_utils import (
    ConfidenceConfig,
    ConfidenceMethodConfig,
    get_confidence_aggregation_bank,
)
from nemo.collections.asr.parts.utils.transcribe_utils import setup_model
from nemo.core.config import hydra_runner
from nemo.utils import logging


@dataclass
class EnsembleConfig:
    # .nemo path or pretrained name
    model: str
    # path to the training data manifest (non-tarred)
    training_manifest: str
    # specify to limit the number of training samples
    # 100 is most likely enough, but setting higher default just in case
    max_training_samples: int = 1000
    # specify to provide dev data manifest for HP tuning
    dev_manifest: Optional[str] = None


@dataclass
class BuildEnsembleConfig:
    # where to save the resulting ensemble model
    output_path: str

    # each model specification
    ensemble: List[EnsembleConfig]

    random_seed: int = 0  # for reproducibility

    # default confidence, can override
    confidence: ConfidenceConfig = ConfidenceConfig(
        # we keep frame confidences and apply aggregation manually to get full-utterance confidence
        preserve_frame_confidence=True,
        exclude_blank=True,
        aggregation="mean",
        method_cfg=ConfidenceMethodConfig(
            name="entropy",
            entropy_type="renui",
            temperature=0.25,  # this is not really temperature, but alpha TODO: should ideally rename + add real temp
            entropy_norm="lin",
        ),
    )

    # this is optional, but can be used to change any aspect of the transcription
    # config, such as batch size or amp usage. Note that model, data and confidence
    # will be overriden by this script
    transcription: TranscriptionConfig = TranscriptionConfig()


def calculate_score(features, labels, pipe):
    """Score is always calculated as mean of the per-class scores.

    This is done to account for possible class imbalances.
    """
    predictions = pipe.predict(features)
    conf_m = confusion_matrix(labels, predictions)
    score = np.diag(conf_m).sum() / conf_m.sum()
    return score, conf_m


def train_model_selection(
    training_features,
    training_labels,
    multi_class="multinomial",
    C=10000.0,  # disabling regularization by default as overfitting is likely not an issue
    class_weight="balanced",  # in case training data is imbalanced
    max_iter=1000,
):
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(multi_class=multi_class, C=C, max_iter=max_iter, class_weight=class_weight),
    )
    pipe.fit(training_features, training_labels)

    accuracy, confusion = calculate_score(training_features, training_labels, pipe)

    logging.info("Training fit accuracy: %.4f", accuracy * 100.0)
    logging.info("Training confusion matrix:\n%s", str(confusion))
    return pipe


# TODO: maybe use parallel transcribe instead?


# TODO: change name?
@hydra_runner(config_name="ensemble_config.yaml", schema=BuildEnsembleConfig)
def main(cfg: BuildEnsembleConfig):
    logging.info(f'Build ensemble config: {OmegaConf.to_yaml(cfg)}')

    # TODO: does this validate arguments? Do we need the check?
    if is_dataclass(cfg):
        cfg = OmegaConf.structured(cfg)

    pl.seed_everything(cfg.random_seed)
    cfg.transcription.random_seed = None  # seed is already applied
    cfg.transcription.ctc_decoding.confidence_cfg = cfg.confidence
    cfg.transcription.rnnt_decoding.confidence_cfg = cfg.confidence

    aggregations = get_confidence_aggregation_bank()
    aggr_func = aggregations[cfg.confidence.aggregation]

    confidences = []
    labels = []
    # note that we loop over the same config. This is intentional, as we need
    # to run all models on all datasets
    for model_idx, model_cfg in enumerate(cfg.ensemble):
        model_confidences = []
        for data_idx, data_cfg in enumerate(cfg.ensemble):
            cfg.transcription.max_samples = model_cfg.max_training_samples
            if model_cfg.model.endswith(".nemo"):
                cfg.transcription.model_path = model_cfg.model
            else:  # assuming pretrained model
                cfg.transcription.pretrained_name = model_cfg.model

            cfg.transcription.dataset_manifest = data_cfg.training_manifest

            with tempfile.NamedTemporaryFile() as output_file:
                cfg.transcription.output_filename = output_file.name
                transcriptions = transcribe_speech(cfg.transcription)

                for transcription in transcriptions:
                    if isinstance(transcription.frame_confidence[0], list):
                        # NeMo Transducer API returns list of lists for confidences
                        conf_values = [conf_value for confs in transcription.frame_confidence for conf_value in confs]
                    else:
                        conf_values = transcription.frame_confidence
                    model_confidences.append(aggr_func(conf_values))
                    if model_idx == 0:  # labels are the same for all models
                        labels.append(data_idx)

        confidences.append(model_confidences)

    # transposing with zip(*list)
    training_features = np.array(list(zip(*confidences)))
    training_labels = np.array(labels)
    model_selection_block = train_model_selection(training_features, training_labels)
    with tempfile.TemporaryDirectory() as tmpdir:
        model_selection_block_path = os.path.join(tmpdir, 'model_selection_block.pkl')
        joblib.dump(model_selection_block, model_selection_block_path)

        # creating ensemble checkpoint
        ensemble_model = ConfidenceEnsembleModel(
            cfg=DictConfig({}),
            trainer=None,
            models=[model_cfg.model for model_cfg in cfg.ensemble],
            model_selection_block_path=model_selection_block_path,
        )
        ensemble_model.save_to(cfg.output_path)

    # ensemble_cfg = {'models': []}
    # ensemble_state = {}
    # for model_idx, model_cfg in enumerate(cfg.ensemble):
    #     model_dict = {}
    #     if model_cfg.model.endswith(".nemo"):
    #         model_dict['model_path'] = model_cfg.model
    #     else:  # assuming pretrained model
    #         model_dict['pretrained_name'] = model_cfg.model
    #     model, _ = setup_model(DictConfig(model_dict), "cpu")
    #     ensemble_cfg['models'].append(model.cfg)
    #     # TODO: can this be done automatically somehow?
    #     for key, value in model.state_dict().items():
    #         ensemble_state[f'models.{model_idx}.{key}'] = value

    # import yaml
    # with open("model_config.yaml", "wt", encoding="utf-8") as fout:
    #     yaml.dump(ensemble_cfg, fout)

    # torch.save(ensemble_state, "model_weights.ckpt")


if __name__ == '__main__':
    main()
