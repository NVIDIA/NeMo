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
"""Utilities for generating text."""

import json
import threading

import torch
from flask import Flask, jsonify, request
from flask_restful import Api, Resource
import torch.nn.functional as F
import torch.distributed as dist

from nemo.collections.nlp.modules.common.retro_inference_strategies import (
    RetroModelTextGenerationStrategy,
    RetroQAModelTextGenerationStrategy,
)
# from nemo.collections.nlp.modules.common.text_generation_utils import generate
from nemo.collections.nlp.modules.common.text_generation_utils_pad import generate
from nemo.utils import logging

GENERATE_NUM = 0
lock = threading.Lock()

STRING_SPK_TOKENS = [str(k) for k in range(10)]

API_ALLOWED_KEYS = set(
    [
        'compute_logprob',
        'all_probs',
        'sentences',
        "task_ids",
        "tokens_to_generate",
        "temperature",
        "add_BOS",
        "greedy",
        "top_k",
        "top_p",
        "neighbors",
        "repetition_penalty",
        "min_tokens_to_generate",
        "end_strings",
    ]
)

def int_to_token(index):
    t_dict = {0: 251521,
        1: 251525,
        2: 251527,
        3: 251556,
        4: 251564,
        5: 251557,
        6: 251574,
        7: 251581,
        8: 251577,
        9: 251561}
    return t_dict[index]

def find_end_from_right(token_ids, end_tokens_list=[982, 426, 251642]):
    index = len(token_ids) - 1
    found = False
    sequence = end_tokens_list
    while index >= 2:  # We check until 2 because we're looking for a sequence of 3
        if token_ids[index] == sequence[2] and token_ids[index-1] == sequence[1] and token_ids[index-2] == sequence[0]:
            found = True
            break
        index -= 1
    if found:
        return index
    else:
        return -1

def rindex(lst, value):
    if value not in lst:
        return -1
    rev_list= lst[::-1]
    i = rev_list.index(value)
    return len(rev_list) - i - 1


def get_ngram_probs(sentences, model, response, target_index_from_end=1):
    if response['full_logprob'] is None:
        raise ValueError("response['full_logprob'] is None. Abort.")
        
    for k in range(len(response['full_logprob'])):
        target_word = sentences[k].split()[-1*target_index_from_end]
        token_id = model.tokenizer.text_to_ids(target_word)[0]
        ridx = rindex(response['token_ids'][k], token_id)
        idx_from_end = len(response['token_ids'][k]) - ridx
        probs = F.softmax(response['full_logprob'][k][-1*idx_from_end], dim=-1)
        if False: 
            vals, idxs = torch.topk(probs, 10)
            bub_string = model.tokenizer.ids_to_text(idxs.tolist())
            word_list = bub_string.split()
            for top_word in word_list:
                print(f"{response['tokens'][k][ridx-5:ridx]}: {top_word}")
        response['word_probs'].append(probs[token_id].item())
        # import ipdb; ipdb.set_trace()
    return response

def get_speaker_probs(sentences, model, response, num_of_speakers=5, cal_word_probs=True):
    for k in range(len(response['full_logprob'])):
        # Find the '▁speaker'(17595) or 'speaker'(211466) token and get the probabilities of the next token
        ridx_nub = rindex(response['token_ids'][k], 211466)
        ridx_wub = rindex(response['token_ids'][k], 17595)
        
         
        if cal_word_probs: 
            end_tokens_list = [982, 426, 251642] # '▁[', 'end', ']'
            the_next_word_is_sym = [3346, 339, 379] # '▁word', '▁is', '('
            end_of_trans_idx_last = find_end_from_right(response['token_ids'][k], end_tokens_list=end_tokens_list)
            end_of_trans_idx = end_of_trans_idx_last - len(end_tokens_list) + 1
            word_idx_anchor = find_end_from_right(response['token_ids'][k], end_tokens_list=the_next_word_is_sym)
            next_word_idx = word_idx_anchor + 1
            next_word_token_id = response['token_ids'][k][next_word_idx]
            # print(f"end_of_trans_idx id : {end_of_trans_idx} end tokens { response['tokens'][k][end_of_trans_idx:end_of_trans_idx+3]}")
            # print(f"Check next word index : {response['tokens'][k][next_word_idx-3:next_word_idx+3]}")
            # print(f"next_word_token_id: {next_word_token_id} next word: ({ model.tokenizer.ids_to_text([next_word_token_id])})")
            probs = F.softmax(response['full_logprob'][k][end_of_trans_idx], dim=-1)
            response['word_probs'].append(probs[next_word_token_id].item())
            # import ipdb; ipdb.set_trace()
        
        ridx = max(ridx_nub, ridx_wub)
        spk_id = response['tokens'][k][ridx+1] 
        if response['token_ids'][k][ridx] not in [211466, 17595]:
            # There is no speaker token in the sentence: 
            logging.info(f"[WARNING] No speaker token found -- ridx: {ridx} token: {response['tokens'][k][ridx]}")
            probs = torch.tensor([(1/num_of_speakers) for q in range(num_of_speakers)])
        else:
            # if ridx == -1 or spk_id not in STRING_SPK_TOKENS:
            #     # token for ridx+1 index is not a number (speaker token)
            #     logging.info(f"[WARNING] Not a number: speaker number token found -- ridx+1: {ridx+1} token: {response['tokens'][k][ridx+1]}")
            #     pass
            idx_from_end = len(response['token_ids'][k]) - (ridx + 1)
            # full_logprob is shifted 1 to the left (first token does not have a probability)
            probs = F.softmax(response['full_logprob'][k][-idx_from_end], dim=-1)
            probs = torch.tensor([probs[int_to_token(q)] for q in range(num_of_speakers)])
        probs_tensor = probs / probs.sum()
        probs_tensor = probs_tensor.numpy()
        response['spk_probs'].append(probs_tensor.tolist())
        # import ipdb; ipdb.set_trace()
    return response 


class MegatronGenerate(Resource):
    def __init__(self, model, inference_strategy=None):
        self.model = model
        self.inference_strategy = inference_strategy

    @staticmethod
    def send_do_generate():
        choice = torch.cuda.LongTensor([GENERATE_NUM])
        torch.distributed.broadcast(choice, 0)

    def put(self):
        # logging.info("request IP: " + str(request.remote_addr))
        # logging.info(json.dumps(request.get_json()))
        # check keys
        for key in request.get_json().keys():
            if key not in API_ALLOWED_KEYS:
                logging.error(f"The request key {key} is not allowed")

        sentences = request.get_json()["sentences"]
        if isinstance(sentences, tuple):  # Input can be text or tensor
            if len(sentences[0]) != len(sentences[1]) or sentences[0] > 128:
                return "Maximum number of sentences is 128", 400
        elif len(sentences) > 128:
            return "Maximum number of sentences is 128", 400

        task_ids = None  # Used for ptuned/prompt tuned models only
        if "task_ids" in request.get_json():
            task_ids = request.get_json()["task_ids"]
            if not isinstance(sentences, tuple):
                return "Input at 'sentences' must by a tuple of two tensors like:\
                    (context_tokens_tensor, context_length_tensor) if task ids are given"
            if len(task_ids) != len(sentences[0]):
                return "Each sentence must have a corresponding task id for p-tuned/prompt-tuned models"

        tokens_to_generate = 64  # Choosing hopefully sane default.  Full sequence is slow
        if "tokens_to_generate" in request.get_json():
            tokens_to_generate = request.get_json()["tokens_to_generate"]
            if not isinstance(tokens_to_generate, int):
                return "tokens_to_generate must be an integer greater than 0"
            if tokens_to_generate < 1:
                return "tokens_to_generate must be an integer greater than 0"

        all_probs = False
        if "all_probs" in request.get_json():
            all_probs = request.get_json()["all_probs"]
            if not isinstance(all_probs, bool):
                return "all_probs must be a boolean value"

        temperature = 1.0
        if "temperature" in request.get_json():
            temperature = request.get_json()["temperature"]
            if not (type(temperature) == int or type(temperature) == float):
                return "temperature must be a positive number less than or equal to 100.0"
            if not (0.0 < temperature <= 100.0):
                return "temperature must be a positive number less than or equal to 100.0"

        add_BOS = False
        if "add_BOS" in request.get_json():
            add_BOS = request.get_json()["add_BOS"]
            if not isinstance(add_BOS, bool):
                return "add_BOS must be a boolean value"

        greedy = False
        if "greedy" in request.get_json():
            greedy = request.get_json()["greedy"]
            if not isinstance(greedy, bool):
                return "greedy must be a boolean value"

        top_k = 0
        if "top_k" in request.get_json():
            top_k = request.get_json()["top_k"]
            if not (type(top_k) == int or type(top_k) == float):
                return "top_k must be a positive integer number"
            if not (0 <= top_k):
                return "top_k must be a positive integer number"

        top_p = 0.9
        if "top_p" in request.get_json():
            top_p = request.get_json()["top_p"]
            if not (type(top_p) == int or type(top_p) == float):
                return "top_p must be a positive number less than or equal to 1.0"
            if not (0.0 <= top_p <= 1.0):
                return "top_p must be a positive number less than or equal to 1.0"

        repetition_penalty = 1.2
        if "repetition_penalty" in request.get_json():
            repetition_penalty = request.get_json()["repetition_penalty"]
            if not (type(repetition_penalty) == int or type(repetition_penalty) == float):
                return "repetition_penalty must be a positive number no less than 1.0"
            if not (1.0 <= repetition_penalty):
                return "repetition_penalty must be a positive number no less than 1.0"

        end_strings = ['<|endoftext|>']
        if 'end_strings' in request.get_json():
            end_strings = request.get_json()['end_strings']
            if not isinstance(end_strings, list):
                return "expect end_strings to be a list of strings"
            if not all([isinstance(s, str) for s in end_strings]):
                return "expect end_strings to be a list of strings"

        min_tokens_to_generate = 0
        if "min_tokens_to_generate" in request.get_json():
            min_tokens_to_generate = request.get_json()["min_tokens_to_generate"]
            if not isinstance(min_tokens_to_generate, int):
                return "min_tokens_to_generate must be an integer no less than 0"
            if min_tokens_to_generate < 0:
                return "min_tokens_to_generate must be an integer no less than 0"

        neighbors = None
        if "neighbors" in request.get_json():
            neighbors = request.get_json()["neighbors"]
            if not isinstance(neighbors, int):
                return "num of neighbors must be an integer no less than 0"
            if neighbors < 0:
                return "num of neighbors must be an integer no less than 0"

        # with lock:  # Need to get lock to keep multiple threads from hitting code
        if True:
            MegatronGenerate.send_do_generate()  # Tell other ranks we're doing generate
            extra = {}
            if task_ids is not None:
                extra['task_ids'] = task_ids
            if self.inference_strategy is not None:
                extra['strategy'] = self.inference_strategy
                # RETRO specific arguments
                if isinstance(
                    self.inference_strategy, (RetroModelTextGenerationStrategy, RetroQAModelTextGenerationStrategy)
                ):
                    if neighbors is not None:
                        self.inference_strategy.update_neighbors(neighbors)
               
            if request.get_json()["tokens_to_generate"] <= 1:
                is_word_probs = True
            else:
                is_word_probs = False
                tokens_to_generate = 1
                
            output = generate(
                self.model,
                sentences,
                tokens_to_generate,
                all_probs,
                temperature,
                add_BOS,
                top_k,
                top_p,
                greedy,
                repetition_penalty,
                compute_logprob=request.get_json()["compute_logprob"],
                end_strings=end_strings,
                min_tokens_to_generate=min_tokens_to_generate,
                **extra,
            )
            # if request.get_json()["tokens_to_generate"] <= 1:
            # if is_word_probs:
            #     output.update({'word_probs': []})
            #     output = get_ngram_probs(request.get_json()["sentences"], self.model, output, target_index_from_end=1)
            # else:
            #     output.update({'spk_probs': []})
            #     output.update({'word_probs': []})
            #     output = get_speaker_probs(request.get_json()["sentences"], self.model, output, num_of_speakers=4)
            output.update({'spk_probs': []})
            output.update({'word_probs': []})
            output = get_speaker_probs(request.get_json()["sentences"], self.model, output, num_of_speakers=4)
            
             
            # for keys in ['sentences', 'token_ids', 'tokens', 'full_logprob', 'logprob', 'offsets']:
            for keys in ['token_ids', 'tokens', 'full_logprob', 'logprob', 'offsets']:
                del output[keys]
            torch.cuda.empty_cache()
            # output['task_ids'] = task_ids
            
            for k in output:
                if isinstance(output[k], torch.Tensor):
                    output[k] = output[k].tolist()

        return jsonify(output)


class MegatronServer(object):
    def __init__(self, model, inference_strategy=None):
        self.app = Flask(__name__, static_url_path='')
        api = Api(self.app)
        api.add_resource(MegatronGenerate, '/generate', resource_class_args=[model, inference_strategy])

    def run(self, url, port=5000):
        self.app.run(url, threaded=True, port=port, debug=False)
