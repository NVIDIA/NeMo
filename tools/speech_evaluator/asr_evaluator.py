# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

from omegaconf import OmegaConf

from nemo.collections.asr.parts.utils.eval_utils import cal_write_wer, run_asr_inference, target_metadata_wer
from nemo.core.config import hydra_runner
from nemo.utils import logging


"""
This script serves as evaluator of ASR models
Usage:
  python python asr_evaluator.py \
asr_eval.pretrained_name="stt_en_conformer_transducer_large" \
asr_eval.inference_mode.mode="offline" \
.....

Check out parameters in ./conf/eval.yaml
"""


@hydra_runner(config_path="conf", config_name="eval.yaml")
def main(cfg):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    report = {}
    # """
    # engine
    ## comment this to use generated manifest
    # cfg.asr_eval = run_asr_inference(cfg.asr_eval)
    # """
    # analyst

    cfg, total_res = cal_write_wer(cfg)
    report.update({"res": total_res})

    for target in cfg.analyst.metadata:
        if cfg.analyst.metadata[target].exec:
            occ_avg_wer = target_metadata_wer(
                cfg.analyst.metric_calculator.output_filename, target, cfg.analyst.metadata[target]
            )
            report[target] = occ_avg_wer

    config_asr_eval = OmegaConf.to_object(cfg.asr_eval)
    report.update(config_asr_eval)

    config_metric_calculator = OmegaConf.to_object(cfg.analyst.metric_calculator)
    report.update(config_metric_calculator)

    pretty = json.dumps(report, indent=4)
    print("===========")
    print(pretty)

    # writer
    report_file = "report.json"
    if "report_filename" in cfg.writer and cfg.writer.report_filename:
        report_file = cfg.writer.report_filename

    with open(report_file, "a") as fout:
        json.dump(report, fout)
        fout.write('\n')
        fout.flush()


if __name__ == "__main__":
    main()
