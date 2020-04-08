# =============================================================================
# Copyright 2020 NVIDIA. All Rights Reserved.
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
# =============================================================================

from nemo import logging

__all__ = ['eval_iter_callback', 'eval_epochs_done_callback']


def eval_iter_callback(tensors, global_vars):
    if "eval_start_logits" not in global_vars.keys():
        global_vars["eval_start_logits"] = []
    if "eval_end_logits" not in global_vars.keys():
        global_vars["eval_end_logits"] = []
    if "eval_unique_ids" not in global_vars.keys():
        global_vars["eval_unique_ids"] = []
    for kv, v in tensors.items():

        if 'logits' in kv:
            logits = []
            for v_tensor in v:
                for logit_tensor in v_tensor:
                    logits.append(logit_tensor.detach().cpu().tolist())

            if kv.startswith('start_logits'):
                global_vars['eval_start_logits'].extend(logits)
            elif kv.startswith('end_logits'):
                global_vars['eval_end_logits'].extend(logits)

        if 'unique_ids' in kv:
            unique_ids = []
            for v_tensor in v:
                for id_tensor in v_tensor:
                    unique_ids.append(id_tensor.detach().cpu().tolist())
            global_vars['eval_unique_ids'].extend(unique_ids)


def eval_epochs_done_callback(
    global_vars,
    eval_data_layer,
    do_lower_case,
    n_best_size,
    max_answer_length,
    version_2_with_negative,
    null_score_diff_threshold,
):
    exact_match, f1, _, _ = eval_data_layer.dataset.evaluate(
        unique_ids=global_vars["eval_unique_ids"],
        start_logits=global_vars["eval_start_logits"],
        end_logits=global_vars["eval_end_logits"],
        n_best_size=n_best_size,
        max_answer_length=max_answer_length,
        version_2_with_negative=version_2_with_negative,
        null_score_diff_threshold=null_score_diff_threshold,
        do_lower_case=do_lower_case,
    )

    logging.info(f"Exact_match = {exact_match}, f1 = {f1}")

    global_vars["eval_unique_ids"] = []
    global_vars["eval_start_logits"] = []
    global_vars["eval_end_logits"] = []

    return dict({"exact_match": exact_match, "f1": f1})
