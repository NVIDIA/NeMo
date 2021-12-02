#!/usr/bin/env python
"""
Augment text by corrupting words in a human-like manner.
"""

import os
import sys
from tqdm import tqdm
from argparse import ArgumentParser
import numpy as np

# TODO: add drop word

#=============================================================================#
# Augmentations
#=============================================================================#


def aug_switch_near_letters(word, p=0.0):
    """
    Switch two consecutive letters in a word
    """
    if (np.random.rand() < p):
        if len(word) > 1:
            i = np.random.randint(len(word)-1)
            j = i+1

            word = (word[:i] + word[j] + word[i] + word[j+1:])

    return word


def aug_drop_letter(word, p=0.0):
    """
    Switch two consecutive letters in a word
    """
    if np.random.rand() < p:
        if len(word) > 1:
            i = np.random.randint(len(word))

            word = (word[:i] + word[i+1:])

    return word

#=============================================================================#
# Main
#=============================================================================#


def main():
    parser = ArgumentParser()
    parser.add_argument("--source", type=str, required=True,
                        help="Input file")
    parser.add_argument("--target", type=str, default="",
                        help="Output file")
    parser.add_argument("--switch_near_letters_order", type=float, default=0.0,
                        help="Rate of switching two consecutive letters in a word")
    parser.add_argument("--drop_letter", type=float, default=0.0,
                        help="Rate of dropping a letter in a word")

    args = parser.parse_args()

    # collect ops
    ops = []
    if args.switch_near_letters_order > 0:
        ops.append(lambda w: aug_switch_near_letters(
            w,
            p=args.switch_near_letters_order))
    if args.drop_letter > 0:
        ops.append(lambda w: aug_drop_letter(
            w,
            p=args.drop_letter))

    with open(args.target, 'w') as target_f:
        for line in open(args.source).readlines():
            line = line.strip()
            words = line.split(" ")
            for op in ops:
                words = list(map(op, words))
            # clean double spaces from dropped words
            line = " ".join(words).replace("  ", " ")

            target_f.write(line + "\n")


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
