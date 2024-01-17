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
#

from nemo.utils import logging


def merge_alignment_with_wb_hyps(
    candidate,
    asr_model,
    cb_results,
    decoder_type="ctc"
):
    

    if decoder_type == "ctc":
        alignment_per_frame = candidate
        # get words borders
        alignment_tokens = []
        prev_token = None
        for idx, token in enumerate(alignment_per_frame):
            if token != asr_model.decoder.blank_idx:
                if token == prev_token:
                    alignment_tokens[-1] = [idx, asr_model.tokenizer.ids_to_tokens([int(token)])[0]]
                else:
                    alignment_tokens.append([idx, asr_model.tokenizer.ids_to_tokens([int(token)])[0]])
            prev_token = token
        
    elif decoder_type == "rnnt":
        alignment_tokens = []
        tokens = asr_model.tokenizer.ids_to_tokens(candidate.y_sequence.tolist())
        for idx, token in enumerate(tokens):
            alignment_tokens.append([candidate.timestep[idx], token])

    if not alignment_tokens:
        for wb_hyp in cb_results:
            pass
            # print(f"wb_hyp: {wb_hyp.word}")
        return " ".join([wb_hyp.word for wb_hyp in cb_results])


    slash = "▁"
    word_alignment = []
    word = ""
    l, r, = None, None
    for item in alignment_tokens:
        if not word:
            word = item[1][1:]
            l = item[0]
            r = item[0]
        else:
            if item[1].startswith(slash):
                word_alignment.append((word, l, r))
                word = item[1][1:]
                l = item[0]
                r = item[0]
            else:
                word += item[1]
                r = item[0]
    word_alignment.append((word, l, r))
    ref_text = [item[0] for item in word_alignment]
    ref_text = " ".join(ref_text)
    # print(f"rnnt_word_alignment: {word_alignment}")

    # merge wb_hyps and word alignment:

    for wb_hyp in cb_results:

        # extend wb_hyp:
        if wb_hyp.start_frame > 0:
            wb_hyp.start_frame -= 1

        new_word_alignment = []
        already_inserted = False
        # lh, rh = wb_hyp.start_frame, wb_hyp.end_frame
        wb_interval = set(range(wb_hyp.start_frame, wb_hyp.end_frame+1))
        for item in word_alignment:
            li, ri = item[1], item[2]
            item_interval = set(range(item[1], item[2]+1))
            if wb_hyp.start_frame < li:
                if not already_inserted:
                    new_word_alignment.append((wb_hyp.word, wb_hyp.start_frame, wb_hyp.end_frame))
                    already_inserted = True

            intersection_part = 100/len(item_interval) * len(wb_interval & item_interval)
            if intersection_part <= 50:
                new_word_alignment.append(item)
            elif not already_inserted:
                new_word_alignment.append((wb_hyp.word, wb_hyp.start_frame, wb_hyp.end_frame))
                already_inserted = True
        # insert last wb word:
        if not already_inserted:
            new_word_alignment.append((wb_hyp.word, wb_hyp.start_frame, wb_hyp.end_frame))

        word_alignment = new_word_alignment
        # print(f"wb_hyp: {wb_hyp.word:<10} -- ({wb_hyp.start_frame}, {wb_hyp.end_frame})")

    boosted_text_list = [item[0] for item in new_word_alignment]
    boosted_text = " ".join(boosted_text_list)
    # print(f"before: {ref_text}")
    # print(f"after : {boosted_text}")
    
    return boosted_text