#!/usr/bin/env python
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
"""
Augment text by corrupting words in a human-like manner.
Support letetrs swap/drop, and AugLy <https://github.com/facebookresearch/AugLy>.
"""

from argparse import ArgumentParser

import numpy as np

try:
    import augly.text as txtaugs
except Exception as e:
    txtaugs = None

# =============================================================================#
# Augmentations
# =============================================================================#


def aug_switch_near_letters(word, p=0.0):
    """
    Switch two consecutive letters in a word
    """
    if np.random.rand() < p:
        if len(word) > 1:
            i = np.random.randint(len(word) - 1)
            j = i + 1

            word = word[:i] + word[j] + word[i] + word[j + 1 :]

    return word


def aug_drop_letter(word, p=0.0):
    """
    Switch two consecutive letters in a word
    """
    if np.random.rand() < p:
        if len(word) > 1:
            i = np.random.randint(len(word))

            word = word[:i] + word[i + 1 :]

    return word


# =============================================================================#
# Main
# =============================================================================#


def main():
    parser = ArgumentParser()
    parser.add_argument("--source", type=str, required=True, help="Input file")
    parser.add_argument("--target", type=str, required=True, help="Output file")
    parser.add_argument(
        "--p_switch_near_letters_order",
        type=float,
        default=0.0,
        help="Probability of switching two consecutive letters in a word",
    )
    parser.add_argument("--p_drop_letter", type=float, default=0.0, help="Probability of dropping a letter in a word")
    # AugLy
    parser.add_argument(
        "--p_augly", type=float, default=0.0, help="Probability of augly to apply a transformation (per word)"
    )

    args = parser.parse_args()

    if (args.p_augly > 0) and (txtaugs is None):
        raise ImportError("Cannot use AugLy, module failed to import. Did you install it? (pip install augly)")

    # collect ops
    ops = []
    if args.p_switch_near_letters_order > 0:
        ops.append(lambda w: aug_switch_near_letters(w, p=args.p_switch_near_letters_order))
    if args.p_drop_letter > 0:
        ops.append(lambda w: aug_drop_letter(w, p=args.p_drop_letter))

    # apply ops
    with open(args.target, 'w') as target_f:
        for line in open(args.source).readlines():
            line = line.strip()
            words = line.split(" ")
            for op in ops:
                words = list(map(op, words))
            # clean double spaces from dropped words
            line = " ".join(words).replace("  ", " ")

            if args.p_augly > 0:
                line = txtaugs.simulate_typos(
                    [line], aug_char_p=args.p_augly, aug_word_p=args.p_augly, aug_char_min=0, aug_word_min=0,
                )[0]

            target_f.write(line + "\n")


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
