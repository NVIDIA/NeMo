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

import torch
from utils.constants import V_NEGATIVE_NUM

import torch
from typing import List, Union


class monitor_cuda_mem:
    _CONTEXT_DEPTH = 0
    ENABLED = True
    EMPTY = False

    def __init__(self, scope, empty=None, enabled: bool = None, precision: int = 4):
        self.scope = scope
        self.empty = empty if empty is not None else monitor_cuda_mem.EMPTY
        self.enabled = enabled if enabled is not None else monitor_cuda_mem.ENABLED
        self.precision = precision

    def __enter__(self):
        monitor_cuda_mem._CONTEXT_DEPTH += 1

        if self.enabled:
            self.print_pad()
            print(f"|> {self.scope}")

            self.initial_memory = torch.cuda.memory_allocated(0)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enabled:
            if self.empty:
                torch.cuda.empty_cache()

            self.final_memory = torch.cuda.memory_allocated(0)

            memory_diff = HumanBytes.format(self.final_memory - self.initial_memory, precision=self.precision)
            self.print_pad()
            print(f"{self.scope} |> {memory_diff}")

        monitor_cuda_mem._CONTEXT_DEPTH -= 1

    @classmethod
    def print_pad(cls):
        print('\t' * (cls._CONTEXT_DEPTH - 1), end='')


# Shortened form of the answer from https://stackoverflow.com/a/63839503
class HumanBytes:
    # fmt: off
    METRIC_LABELS: List[str] = ["B", "kB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"]
    BINARY_LABELS: List[str] = ["B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB", "ZiB", "YiB"]
    PRECISION_OFFSETS: List[float] = [5 * (0.1 ** x) for x in range(1, 22)]  # PREDEFINED FOR SPEED.
    PRECISION_FORMATS: List[str] = [("{}{:." + str(ratio) + "f} {}") for ratio in range(len(PRECISION_OFFSETS))]  # PREDEFINED FOR SPEED.

    # fmt: on

    @staticmethod
    def format(num: Union[int, float], metric: bool = False, precision: int = 1) -> str:
        assert isinstance(num, (int, float)), "num must be an int or float"
        assert isinstance(metric, bool), "metric must be a bool"
        assert (
            isinstance(precision, int) and precision >= 0 and precision <= len(HumanBytes.PRECISION_OFFSETS)
        ), "precision must be an int (range 0-20)"

        unit_labels = HumanBytes.METRIC_LABELS if metric else HumanBytes.BINARY_LABELS
        last_label = unit_labels[-1]
        unit_step = 1000 if metric else 1024
        unit_step_thresh = unit_step - HumanBytes.PRECISION_OFFSETS[precision]

        is_negative = num < 0
        if is_negative:  # Faster than ternary assignment or always running abs().
            num = abs(num)

        for unit in unit_labels:
            if num < unit_step_thresh:
                break
            if unit != last_label:
                num /= unit_step

        return HumanBytes.PRECISION_FORMATS[precision].format("-" if is_negative else "", num, unit)


@torch.no_grad()
@torch.inference_mode()
def viterbi_decoding(log_probs_batch, y_batch, T_batch, U_batch, viterbi_device):
    """
    Do Viterbi decoding with an efficient algorithm (the only for-loop in the 'forward pass' is over the time dimension). 
    Args:
        log_probs_batch: tensor of shape (B, T_max, V). The parts of log_probs_batch which are 'padding' are filled
            with 'V_NEGATIVE_NUM' - a large negative number which represents a very low probability.
        y_batch: tensor of shape (B, U_max) - contains token IDs including blanks in every other position. The parts of
            y_batch which are padding are filled with the number 'V'. V = the number of tokens in the vocabulary + 1 for
            the blank token.
        T_batch: tensor of shape (B, 1) - contains the durations of the log_probs_batch (so we can ignore the 
            parts of log_probs_batch which are padding)
        U_batch: tensor of shape (B, 1) - contains the lengths of y_batch (so we can ignore the parts of y_batch
            which are padding).
        viterbi_device: the torch device on which Viterbi decoding will be done.

    Returns:
        alignments_batch: list of lists containing locations for the tokens we align to at each timestep.
            Looks like: [[0, 0, 1, 2, 2, 3, 3, ...,  ], ..., [0, 1, 2, 2, 2, 3, 4, ....]].
            Each list inside alignments_batch is of length T_batch[location of utt in batch].
    """
    B, T_max, _ = log_probs_batch.shape
    U_max = y_batch.shape[1]

    w
    # transfer all tensors to viterbi_device
    log_probs_batch = log_probs_batch.to(viterbi_device)
    y_batch = y_batch.to(viterbi_device)
    T_batch = T_batch.to(viterbi_device)
    U_batch = U_batch.to(viterbi_device)

    # make tensor that we will put at timesteps beyond the duration of the audio
    padding_for_log_probs = V_NEGATIVE_NUM * torch.ones((B, T_max, 1), device=viterbi_device)
    # make log_probs_padded tensor of shape (B, T_max, V +1 ) where all of
    # log_probs_padded[:,:,-1] is the 'V_NEGATIVE_NUM'
    log_probs_padded = torch.cat((log_probs_batch, padding_for_log_probs), dim=2)
    # make log_probs_reordered tensor of shape (B, T_max, U_max)
    # it contains the log_probs for only the tokens that are in the Ground Truth, and in the order
    # that they occur
    print(log_probs_batch.dtype, log_probs_batch.shape)
    print()
    del log_probs_batch
    index = y_batch.unsqueeze(1).expand(-1, T_max, -1)
    log_probs_padded = log_probs_padded.to(torch.bfloat16)
    log_probs_reordered = torch.gather(input=log_probs_padded, dim=2, index=index)

    torch.cuda.empty_cache()

    print("Log probs reordered", log_probs_reordered.shape, log_probs_reordered.dtype)

    # initialize tensors of viterbi probabilies and backpointers
    v_matrix = V_NEGATIVE_NUM * torch.ones_like(log_probs_reordered)
    backpointers = -999 * torch.ones_like(v_matrix)
    v_matrix[:, 0, :2] = log_probs_reordered[:, 0, :2]

    # Make a letter_repetition_mask the same shape as y_batch
    # the letter_repetition_mask will have 'True' where the token (including blanks) is the same
    # as the token two places before it in the ground truth (and 'False everywhere else).
    # We will use letter_repetition_mask to determine whether the Viterbi algorithm needs to look two tokens back or
    # three tokens back
    y_shifted_left = torch.roll(y_batch, shifts=2, dims=1)
    letter_repetition_mask = y_batch - y_shifted_left
    letter_repetition_mask[:, :2] = 1  # make sure dont apply mask to first 2 tokens
    letter_repetition_mask = letter_repetition_mask == 0

    # bp_absolute_template is a tensor we will need during the Viterbi decoding to convert our argmaxes from indices between 0 and 2,
    # to indices in the range (0, U_max-1) indicating from which token the mostly path up to that point came from.
    # it is a tensor of shape (B, U_max) that looks like
    # bp_absolute_template = [
    #   [0, 1, 2, ...,, U_max]
    #   [0, 1, 2, ...,, U_max]
    #   [0, 1, 2, ...,, U_max]
    #   ... rows repeated so there are B number of rows in total
    # ]
    bp_absolute_template = torch.arange(U_max, device=viterbi_device).unsqueeze(0).repeat(B, 1)

    for t in range(1, T_max):

        # e_current is a tensor of shape (B, U_max) of the log probs of every possible token at the current timestep
        e_current = log_probs_reordered[:, t, :]

        # v_prev is a tensor of shape (B, U_max) of the viterbi probabilities 1 timestep back and in the same token position
        v_prev = v_matrix[:, t - 1, :]

        # v_prev_shifted is a tensor of shape (B, U_max) of the viterbi probabilities 1 timestep back and 1 token position back
        v_prev_shifted = torch.roll(v_prev, shifts=1, dims=1)
        # by doing a roll shift of size 1, we have brought the viterbi probability in the final token position to the
        # first token position - let's overcome this by 'zeroing out' the probabilities in the firest token position
        v_prev_shifted[:, 0] = V_NEGATIVE_NUM

        # v_prev_shifted2 is a tensor of shape (B, U_max) of the viterbi probabilities 1 timestep back and 2 token position back
        v_prev_shifted2 = torch.roll(v_prev, shifts=2, dims=1)
        v_prev_shifted2[:, :2] = V_NEGATIVE_NUM  # zero out as we did for v_prev_shifted
        # use our letter_repetition_mask to remove the connections between 2 blanks (so we don't skip over a letter)
        # and to remove the connections between 2 consective letters (so we don't skip over a blank)
        v_prev_shifted2.masked_fill_(letter_repetition_mask, V_NEGATIVE_NUM)

        # we need this v_prev_dup tensor so we can calculated the viterbi probability of every possible
        # token position simultaneously
        v_prev_dup = torch.cat(
            (v_prev.unsqueeze(2), v_prev_shifted.unsqueeze(2), v_prev_shifted2.unsqueeze(2),), dim=2,
        )

        # candidates_v_current are our candidate viterbi probabilities for every token position, from which
        # we will pick the max and record the argmax
        candidates_v_current = v_prev_dup + e_current.unsqueeze(2)
        v_current, bp_relative = torch.max(candidates_v_current, dim=2)

        # convert our argmaxes from indices between 0 and 2, to indices in the range (0, U_max-1) indicating
        # from which token the mostly path up to that point came from
        bp_absolute = bp_absolute_template - bp_relative

        # update our tensors containing all the viterbi probabilites and backpointers
        v_matrix[:, t, :] = v_current
        backpointers[:, t, :] = bp_absolute

    # trace backpointers TODO: parallelize over batch_size
    alignments_batch = []
    for b in range(B):
        T_b = int(T_batch[b])
        U_b = int(U_batch[b])

        final_state = int(torch.argmax(v_matrix[b, T_b - 1, U_b - 2 : U_b])) + U_b - 2
        alignment_b = [final_state]
        for t in range(T_b - 1, 0, -1):
            alignment_b.insert(0, int(backpointers[b, t, alignment_b[0]]))
        alignments_batch.append(alignment_b)

    return alignments_batch
