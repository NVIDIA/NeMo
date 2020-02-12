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

__all__ = ['eval_iter_callback', 'eval_epochs_done_callback']

GLOBAL_KEYS = ["eval_loss", "ref", "sys", "sent_ids", "nonpad_tokens"]


def eval_iter_callback(tensors, global_vars, tgt_tokenizer):
    for key in GLOBAL_KEYS:
        if key not in global_vars.keys():
            global_vars[key] = []

    for kv, v in tensors.items():

        if "output_ids" in kv:
            sys = []
            for beam in v:
                beam_search_translation = beam.cpu().numpy().tolist()
                for sentence in beam_search_translation:
                    sys.append(tgt_tokenizer.ids_to_text(sentence))
            global_vars["sys"].append(sys)

        if "tgt" in kv:
            ref = []
            for tgt in v:
                nonpad_tokens = (tgt != tgt_tokenizer.pad_id).sum().item()
                tgt_sentences = tgt.cpu().numpy().tolist()
                for sentence in tgt_sentences:
                    ref.append(tgt_tokenizer.ids_to_text(sentence))
                global_vars["nonpad_tokens"].append(nonpad_tokens)
            global_vars["ref"].append(ref)

        if "sent_ids" in kv:
            for sent_ids in v:
                global_vars["sent_ids"].extend(sent_ids.cpu().numpy().tolist())

        if "loss" in kv:
            for eval_loss in v:
                global_vars["eval_loss"].append(eval_loss.item())


def eval_epochs_done_callback(global_vars, validation_dataset=None):
    losses = np.array(global_vars["eval_loss"])
    counts = np.array(global_vars["nonpad_tokens"])
    eval_loss = np.sum(losses * counts) / np.sum(counts)

    all_sys = [j for i in global_vars["sys"] for j in i]
    _, indices = np.unique(global_vars["sent_ids"], return_index=True)
    all_sys = [all_sys[i] for i in indices]

    if validation_dataset is not None:
        all_ref = [open(validation_dataset, "r").readlines()]
        # _, *refs = download_test_set("wmt14/full", "en-de")
        # all_ref = [smart_open(x).readlines() for x in refs]
    else:
        all_ref = [[j for i in global_vars["ref"] for j in i]]

    token_bleu = corpus_bleu(all_sys, all_ref, tokenize="fairseq").score
    sacre_bleu = corpus_bleu(all_sys, all_ref, tokenize="13a").score

    for i in range(3):
        sent_id = np.random.randint(len(all_sys))
        logging.info("Ground truth: {0}\n".format(all_ref[0][sent_id]))
        logging.info("Translation:  {0}\n".format(all_sys[sent_id]))

    logging.info("------------------------------------------------------------")
    logging.info("Validation loss: {0}".format(np.round(eval_loss, 3)))
    logging.info("TokenBLEU: {0}".format(np.round(token_bleu, 2)))
    logging.info("SacreBLEU: {0}".format(np.round(sacre_bleu, 2)))
    logging.info("------------------------------------------------------------")

    for key in GLOBAL_KEYS:
        global_vars[key] = []

    metrics = dict({"eval_loss": eval_loss, "token_bleu": token_bleu, "sacre_bleu": sacre_bleu})

    return metrics


def eval_epochs_done_callback_wer(global_vars):
    eval_loss = np.mean(global_vars["eval_loss"])
    all_ref = []
    for r in global_vars["ref"]:
        all_ref += r
    all_sys = []
    for s in global_vars["sys"]:
        all_sys += s
    ref = all_ref
    sys = all_sys
    eval_wer = word_error_rate(ref, sys)
    for i in range(3):
        sent_id = np.random.randint(len(sys))
        logging.info("Ground truth: {0}\n".format(ref[sent_id]))
        logging.info("Translation:  {0}\n".format(sys[sent_id]))

    logging.info("Validation loss: {0}".format(np.round(eval_loss, 3)))
    logging.info("Validation WER: {0}".format(eval_wer))
    global_vars["eval_loss"] = []
    global_vars["ref"] = []
    global_vars["sys"] = []

    return dict({"eval_loss": eval_loss, "eval_wer": eval_wer})
