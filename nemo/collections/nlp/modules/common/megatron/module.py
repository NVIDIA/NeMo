# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

"""Megatron Module"""

import torch
from apex.transformer import parallel_state, tensor_parallel


def param_is_not_shared(param):
    return not hasattr(param, 'shared') or not param.shared


class MegatronModule(torch.nn.Module):
    """Megatron specific extensions of torch Module with support
    for pipelining."""

    def __init__(self, share_word_embeddings=True):
        super(MegatronModule, self).__init__()
        self.share_word_embeddings = share_word_embeddings

    def word_embeddings_weight(self):
        if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
            return self.language_model.embedding.word_embeddings.weight
        if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
            if not self.share_word_embeddings:
                raise Exception(
                    'word_embeddings_weight() called for last ' 'stage, but share_word_embeddings is false'
                )
            return self.word_embeddings.weight
        raise Exception('word_embeddings_weight() should be ' 'called for first and last stage only')

    def initialize_word_embeddings(self, init_method, vocab_size, hidden_size, pipeline_model_parallel_size=1):
        if not self.share_word_embeddings:
            raise Exception('initialize_word_embeddings() was called but ' 'share_word_embeddings is false')

        # TODO: pipeline model parallelism is not implemented in NeMo yet
        # This function just initializes the word embeddings in the final stage
        # when we are using pipeline parallelism. If we aren't using pipeline
        # parallelism there is nothing to do.
        if pipeline_model_parallel_size == 1:
            return

        # Parameters are shared between the word embeddings layer, and the
        # heads at the end of the model. In a pipelined setup with more than
        # one stage, the initial embedding layer and the head are on different
        # workers, so we do the following:
        # 1. Create a second copy of word_embeddings on the last stage, with
        #    initial parameters of 0.0.
        # 2. Do an all-reduce between the first and last stage to ensure that
        #    the two copies of word_embeddings start off with the same
        #    parameter values.
        # 3. In the training loop, before an all-reduce between the grads of
        #    the two word_embeddings layers to ensure that every applied weight
        #    update is the same on both stages.
        if parallel_state.is_pipeline_last_stage():
            assert not parallel_state.is_pipeline_first_stage()
            self._word_embeddings_for_head_key = 'word_embeddings_for_head'
            # set word_embeddings weights to 0 here, then copy first
            # stage's weights using all_reduce below.
            self.word_embeddings = tensor_parallel.VocabParallelEmbedding(
                vocab_size, hidden_size, init_method=init_method
            )
            self.word_embeddings.weight.data.fill_(0)
            self.word_embeddings.weight.shared = True

        # Ensure that first and last stages have the same initial parameter
        # values.
        if torch.distributed.is_initialized():
            if parallel_state.is_pipeline_first_stage() or parallel_state.is_pipeline_last_stage():
                torch.distributed.all_reduce(
                    self.word_embeddings_weight().data, group=parallel_state.get_embedding_group()
                )
        else:
            print(
                "WARNING! Distributed processes aren't initialized, so "
                "word embeddings in the last layer are not initialized. "
                "If you are just manipulating a model this is fine, but "
                "this needs to be handled manually. If you are training "
                "something is definitely wrong."
            )
