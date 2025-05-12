# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from glob import glob
from typing import List

from omegaconf import MISSING

from nemo.collections.asr.parts.context_biasing.gpu_boosting.context_graph import ContextGraph
from nemo.collections.asr.parts.context_biasing.gpu_boosting.boosting_graph_batched import GPUBoostingTreeModel
from nemo.collections.asr.parts.submodules.ngram_lm import NGramGPULanguageModel, KenLMBatchedWrapper
from nemo.collections.common.tokenizers import AggregateTokenizer
import torch
from torch.nn.utils.rnn import pad_sequence
import nemo.collections.asr as nemo_asr
from nemo.core.config import hydra_runner
import numpy as np

from nemo.utils import logging



@dataclass
class BuildWordBoostingTreeConfig:
    """
    Build GPU-accelerated word boosting graph to be used with greedy and beam search decoders of ASR models.
    """

    asr_model_nemo_file: str = MISSING  # The path to '.nemo' file of the ASR model, or name of a pretrained NeMo model
    context_biasing_list: str = MISSING  # The path to the context-biasing list file.
    path_to_save_btree: str = MISSING  # The path to save the GPU-accelerated word boosting graph.

    context_score: float = 1.0  # The score for each arc transition in the context graph.
    unk_score: float = 0.5  # The score for unknown tokens (tokens that are not in the beginning of context-biasing phrases)
    score_per_phrase: float = 0.0  # Custom score for each phrase in the context graph.

    # alternative transcription generation
    use_bpe_dropout: bool = False # Whether to use BPE dropout for generating alternative transcriptions.
    num_of_transcriptions: int = 5  # The number of alternative transcriptions to generate for each context-biasing phrase.
    bpe_alpha: float = 0.3  # The alpha parameter for BPE dropout.


    test_btree_model: bool = False  # Whether to test the GPU-accelerated word boosting graph after building it.



@hydra_runner(config_path=None, config_name='TrainKenlmConfig', schema=BuildWordBoostingTreeConfig)
def main(cfg: BuildWordBoostingTreeConfig):

    # 1. load asr model for tokenizer
    asr_model = nemo_asr.models.ASRModel.restore_from(cfg.asr_model_nemo_file, map_location=torch.device('cpu'))
    is_aggregate_tokenizer = isinstance(asr_model.tokenizer, AggregateTokenizer)
    source_lang = "en" # TODO: replace hardcode lang

    # 2. tokenize context-biasing phrases
    cb_dict = {}
    with open(cfg.context_biasing_list, "r") as f:
        for line in f:
            line = line.strip()
            if not is_aggregate_tokenizer:
                cb_dict[line] = asr_model.tokenizer.text_to_ids(line)
                if cfg.use_bpe_dropout:
                    cb_dict[line] = [asr_model.tokenizer.text_to_ids(line)]
                    trans_set = set()
                    trans_set.add(" ".join(asr_model.tokenizer.text_to_tokens(line)))
                    i = 1
                    cur_step = 1
                    while i < cfg.num_of_transcriptions and cur_step < cfg.num_of_transcriptions * 2:
                        cur_step += 1                       
                        trans = asr_model.tokenizer.tokenizer.encode(line, enable_sampling=True, alpha=cfg.bpe_alpha, nbest_size=-1)
                        trans_text = asr_model.tokenizer.ids_to_tokens(trans)
                        if trans_text[0] == "▁":
                            continue
                        trans_text = " ".join(trans_text)
                        if trans_text not in trans_set:
                            cb_dict[line].append(trans)
                            trans_set.add(trans_text)
                            i += 1
                        # import ipdb; ipdb.set_trace()
            else:
                cb_dict[line] = asr_model.tokenizer.text_to_ids(line, source_lang)


    # cfg.unk_score = float(round(np.log10(1/len(asr_model.tokenizer.vocab)), 2))
    # cfg.context_score = round(cfg.unk_score/1.5, 2)
    
    
    # 3. build icefall context-biasing graph
    contexts = []
    scores = []
    phrases = []
    for phrase in cb_dict:
        if cfg.use_bpe_dropout:
            # import ipdb; ipdb.set_trace()
            for trans in cb_dict[phrase]:
                contexts.append(trans)
                scores.append(round(cfg.score_per_phrase / len(phrase), 2))
                phrases.append(phrase)
        else:
            contexts.append(cb_dict[phrase])
            scores.append(round(cfg.score_per_phrase / len(phrase), 2))
            phrases.append(phrase)

    context_graph = ContextGraph(context_score=cfg.context_score)
    context_graph.build(token_ids=contexts, scores=scores, phrases=phrases)

    # 4. convert icefall context-biasing graph to gpu boosting tree
    vocab_size = len(asr_model.tokenizer.vocab)
    # import ipdb; ipdb.set_trace()
    if not is_aggregate_tokenizer:
        gpu_boosting_model = GPUBoostingTreeModel.from_cb_tree(context_graph, vocab_size=vocab_size, unk_score=cfg.unk_score, use_triton=False)
    else:
        gpu_boosting_model = GPUBoostingTreeModel.from_cb_tree(context_graph, vocab_size=vocab_size, unk_score=cfg.unk_score, eos_id=asr_model.tokenizer.eos_id, use_triton=False)

    # 5. save gpu boosting tree to nemo file
    gpu_boosting_model.save_to(cfg.path_to_save_btree)

    # 6. test gpu boosting tree model
    logging.info("testing gpu boosting tree model...")
    if cfg.test_btree_model:
        gpu_boosting_model_loaded = GPUBoostingTreeModel.from_nemo(cfg.path_to_save_btree, vocab_size=vocab_size, use_triton=False)
        device = torch.device("cuda")
        gpu_boosting_model_loaded = gpu_boosting_model_loaded.cuda()

        sentences = [
            "hello world",
            "i like nvidia",
            "nvlink",
            "nvlinks two",
            "nvlinz",
            "gpu boosting",
            "lot of gpus",
            "omniverse cloud now",
        ]
        if not is_aggregate_tokenizer:
            sentences_ids = [asr_model.tokenizer.text_to_ids(sentence) for sentence in sentences]
            sentences_tokens = [asr_model.tokenizer.text_to_tokens(sentence) for sentence in sentences]
        else:
            sentences_ids = [asr_model.tokenizer.text_to_ids(sentence, source_lang) for sentence in sentences]
            sentences_ids.append([3])
            # sentences_tokens = [asr_model.tokenizer.text_to_tokens(sentence, source_lang) for sentence in sentences]
            sentences_tokens = []

        scores_lm = gpu_boosting_model_loaded(
            labels=pad_sequence([torch.LongTensor(sentence) for sentence in sentences_ids], batch_first=True).to(device),
            labels_lengths=torch.LongTensor([len(sentence) for sentence in sentences_ids]).to(device),
            bos=False,
            eos=False,
        )

        logging.info(f"[info]: scores_lm: {scores_lm}")
        logging.info(f"[info]: sentences_text: {sentences}")
        logging.info(f"[info]: sentences_text: {sentences_tokens}")
        logging.info(f"[info]: sentences_ids: {sentences_ids}")

    # TODO: move this to a test script
    # # 7. load lm
    # nemo_path="/mnt/ssd2/data/evaluation_sets/nvtalks/context_biasing_lists/oracle.txt.rnnt-lstm.6-order.no_prun.no-norm.lm.nemo"
    # gpu_boosting_model_loaded = NGramGPULanguageModel.from_file(nemo_path, vocab_size=vocab_size)
    # # device = torch.device("cpu")
    # gpu_boosting_model_loaded = gpu_boosting_model_loaded.cuda()

    # sentences = [
    #     "hello world",
    #     "i like nvidia",
    #     "nvlink",
    #     "nvlinks two",
    #     "nvlinz",
    #     "gpu boosting",
    #     "lot of gpus",
    #     "omniverse cloud now",
    # ]
    # sentences_ids = [asr_model.tokenizer.text_to_ids(sentence) for sentence in sentences]
    # sentences_tokens = [asr_model.tokenizer.text_to_tokens(sentence) for sentence in sentences]

    # scores_lm = gpu_boosting_model_loaded(
    #     labels=pad_sequence([torch.LongTensor(sentence) for sentence in sentences_ids], batch_first=True).to(device),
    #     labels_lengths=torch.LongTensor([len(sentence) for sentence in sentences_ids]).to(device),
    #     bos=True,
    #     eos=False,
    # )

    # print(f"[info]: scores_lm: {scores_lm}")
    # print(f"[info]: sentences_text: {sentences}")
    # print(f"[info]: sentences_text: {sentences_tokens}")
    # print(f"[info]: sentences_ids: {sentences_ids}")

    # 8. load kenlm
    # nemo_path="/mnt/ssd2/data/evaluation_sets/nvtalks/context_biasing_lists/oracle.txt.rnnt-lstm.6-order.no_prun.no-norm.lm.tmp.arpa"
    # gpu_boosting_model_loaded = KenLMBatchedWrapper.from_file(nemo_path, vocab_size=vocab_size)
    # device = torch.device("cpu")
    # gpu_boosting_model_loaded = gpu_boosting_model_loaded

    # sentences = [
    #     "hello world",
    #     "i like nvidia",
    #     "nvlink",
    #     "nvlinks two",
    #     "nvlinz",
    #     "gpu boosting",
    #     "lot of gpus",
    # ]
    # sentences_ids = [asr_model.tokenizer.text_to_ids(sentence) for sentence in sentences]
    # sentences_tokens = [asr_model.tokenizer.text_to_tokens(sentence) for sentence in sentences]

    # scores_lm = gpu_boosting_model_loaded.score_sentences(
    #     sentences_ids,
    #     bos=True,
    #     eos=False,
    # )

    # print(f"[info]: scores_lm: {scores_lm}")
    # print(f"[info]: sentences_text: {sentences}")
    # print(f"[info]: sentences_text: {sentences_tokens}")
    # print(f"[info]: sentences_ids: {sentences_ids}")




if __name__ == '__main__':
    main()
