# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Optional

from nemo_run import Config, cli
from pytorch_lightning.callbacks.callback import Callback

from nemo.utils.import_utils import safe_import

res_module, HAVE_RES = safe_import('nvidia_resiliency_ext.ptl_resiliency')


@cli.factory(is_target_default=True)
def straggler_det_callback(
    straggler_report_time_interval: Optional[int] = 300, stop_if_detected_straggler: Optional[bool] = True
) -> Config[res_module.StragglerDetectionCallback]:

    return Config(
        res_module.StragglerDetectionCallback,
        report_time_interval=straggler_report_time_interval,
        calc_relative_gpu_perf=True,
        calc_individual_gpu_perf=True,
        num_gpu_perf_scores_to_print=5,
        gpu_relative_perf_threshold=0.7,
        gpu_individual_perf_threshold=0.7,
        stop_if_detected=stop_if_detected_straggler,
        enable_ptl_logging=True,
    )
