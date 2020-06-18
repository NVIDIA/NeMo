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

import numpy as np

from nemo import logging
from nemo.collections.asr.metrics import word_error_rate
from nemo.collections.nlp.metrics.sacrebleu import corpus_bleu
import pdb
import torch

__all__ = ['eval_iter_callback', 'eval_epochs_done_callback']

GLOBAL_KEYS = [
	"tgt_ids", "input_ids", "loss", "per_example_loss", 
	"log_softmax", "beam_results", "src_ids", "src_first_tokens",
	"pred", "labels", "labels_mask"
]

def eval_iter_callback(tensors, global_vars, tokenizer):
	for key in GLOBAL_KEYS:
		if key not in global_vars.keys():
			global_vars[key] = []

	for kv, v in tensors.items():
		# print(kv)

		# if "output_ids" in kv:
		# 	sys = []
		# 	for beam in v:
		# 		beam_search_translation = beam.cpu().numpy().tolist()
		# 		pdb.set_trace()
		# 		for sentence in beam_search_translation:
		# 			sys.append(tokenizer.ids_to_text(sentence))
		# 	global_vars["beam_reasults"].append(sys)

		# if "tgt" in kv:
		#     ref = []
		#     for tgt in v:
		#         nonpad_tokens = (tgt != tgt_tokenizer.pad_id).sum().item()
		#         tgt_sentences = tgt.cpu().numpy().tolist()
		#         for sentence in tgt_sentences:
		#             ref.append(tgt_tokenizer.ids_to_text(sentence))
		#         global_vars["nonpad_tokens"].append(nonpad_tokens)
		#     global_vars["ref"].append(ref)

		# if "tgt_ids" in kv:
		# 	for tgt_ids in v:
		# 		import pdb
		# 		pdb.set_trace()
		# 		das = tgt_ids.cpu().numpy().tolist()
		# 		# global_vars["tgt_ids"].extend(das)

		# if "crossentropylossnm0" in kv:
		# 	for loss in v:
		# 		global_vars["loss"].extend(loss.cpu().numpy().tolist())

		if "crossentropylossnm1" in kv:
			for per_example_loss in v:
				pel = per_example_loss.cpu().numpy().tolist()
				global_vars["per_example_loss"].extend(pel)

		if "logits" in kv:
			for pred in v:
				p = torch.argmax(pred, dim=-1).int().cpu().numpy().tolist()
				global_vars["pred"].extend(p)

		if "labels~" in kv:
			for label in v:
				l = label.cpu().numpy().tolist()
				global_vars["labels"].extend(l)
		
		if "labels_mask" in kv:
			for mask in v:
				m = mask.cpu().numpy().tolist()
				global_vars["labels_mask"].extend(m)

		# if "src_ids" in kv:
		# 	for src_ids in v:
		# 		pdb.set_trace()
		# 		# global_vars["src_tokens"].append(token)

		# if "src_first_tokens" in kv:
		# 	for first_token_idx in v:
		# 		pdb.set_trace()


def eval_epochs_done_callback(global_vars, validation_dataset=None):
	losses = np.array(global_vars["per_example_loss"])
	eval_loss = np.mean(losses)
	global_vars["per_example_loss"] = []

	labels = np.array([np.array(n) for n in global_vars["labels"]])
	predictions = np.array([np.array(n) for n in global_vars["pred"]])
	labels_mask = np.array([np.array(n) for n in global_vars["labels_mask"]])
	for key in GLOBAL_KEYS:
		global_vars[key] = []

	lor = np.logical_or(labels==predictions, ~labels_mask.astype(np.bool))
	accuracy = np.all(lor, axis=1).astype(np.float32)

	logging.info("------------------------------------------------------------")
	logging.info("Validation loss: {0}".format(np.round(eval_loss, 3)))
	logging.info("Sentence level accuracy: {0}".format(accuracy))
	logging.info("------------------------------------------------------------")

	return dict({"eval_loss": eval_loss})
