# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
This script can be used after GIZA++ alignment to extract final alignments for each semiotic class.
"""

import re
from argparse import ArgumentParser
from typing import List, Tuple

import numpy as np

parser = ArgumentParser(description='Extract final alignments from GIZA++ alignments')
parser.add_argument('--giza_dir', type=str, required=True, help='Path to folder with GIZA++ alignment')
parser.add_argument(
    '--giza_suffix', type=str, required=True, help='suffix of alignment files, e.g. \"Ahmm.5\", \"A3.final\"'
)
parser.add_argument('--out_filename', type=str, required=True, help='Output file')
args = parser.parse_args()


def fill_alignment_matrix(
    fline2: str, fline3: str, gline2: str, gline3: str
) -> Tuple[np.ndarray, List[str], List[str]]:
    """Parse Giza++ direct and reverse alignment results and represent them as an alignment matrix

    Args:
        fline2: e.g. "_2 0 1 4_"
        fline3: e.g. "NULL ({ }) twenty ({ 1 }) fourteen ({ 2 3 4 })"
        gline2: e.g. "twenty fourteen"
        gline3: e.g. "NULL ({ }) _2 ({ 1 }) 0 ({ }) 1 ({ }) 4_ ({ 2 })"

    Returns:
        matrix: a numpy array of shape (src_len, dst_len) filled with [0, 1, 2, 3], where 3 means a reliable alignment
         the corresponding words were aligned to one another in direct and reverse alignment runs, 1 and 2 mean that the
         words were aligned only in one direction, 0 - no alignment.
        srctokens: e.g. ["twenty", "fourteen"]
        dsttokens: e.g. ["_2", "0", "1", "4_"]

    For example, the alignment matrix for the above example may look like:
    [[3, 0, 0, 0]
     [0, 2, 2, 3]]
    """
    if fline2 is None or gline2 is None or fline3 is None or gline3 is None:
        raise ValueError(f"empty params")
    srctokens = gline2.split()
    dsttokens = fline2.split()
    pattern = r"([^ ]+) \(\{ ([^\(\{\}\)]*) \}\)"
    src2dst = re.findall(pattern, fline3.replace("({ })", "({  })"))
    dst2src = re.findall(pattern, gline3.replace("({ })", "({  })"))
    if len(src2dst) != len(srctokens) + 1:
        raise ValueError(
            "length mismatch: len(src2dst)="
            + str(len(src2dst))
            + "; len(srctokens)"
            + str(len(srctokens))
            + "\n"
            + gline2
            + "\n"
            + fline3
        )
    if len(dst2src) != len(dsttokens) + 1:
        raise ValueError(
            "length mismatch: len(dst2src)="
            + str(len(dst2src))
            + "; len(dsttokens)"
            + str(len(dsttokens))
            + "\n"
            + fline2
            + "\n"
            + gline3
        )
    matrix = np.zeros((len(srctokens), len(dsttokens)))
    for i in range(1, len(src2dst)):
        token, to_str = src2dst[i]
        if to_str == "":
            continue
        to = list(map(int, to_str.split()))
        for t in to:
            matrix[i - 1][t - 1] = 2

    for i in range(1, len(dst2src)):
        token, to_str = dst2src[i]
        if to_str == "":
            continue
        to = list(map(int, to_str.split()))
        for t in to:
            matrix[t - 1][i - 1] += 1

    return matrix, srctokens, dsttokens


def check_monotonicity(matrix: np.ndarray) -> bool:
    """Check if alignment is monotonous - i.e. the relative order is preserved (no swaps).

    Args:
        matrix: a numpy array of shape (src_len, dst_len) filled with [0, 1, 2, 3], where 3 means a reliable alignment
         the corresponding words were aligned to one another in direct and reverse alignment runs, 1 and 2 mean that the
         words were aligned only in one direction, 0 - no alignment.
    """
    is_sorted = lambda k: np.all(k[:-1] <= k[1:])

    a = np.argwhere(matrix == 3)
    b = np.argwhere(matrix == 2)
    c = np.vstack((a, b))
    d = c[c[:, 1].argsort()]  # sort by second column (less important)
    d = d[d[:, 0].argsort(kind="mergesort")]
    return is_sorted(d[:, 1])


def get_targets(matrix: np.ndarray, dsttokens: List[str]) -> List[str]:
    """Join some of the destination tokens, so that their number becomes the same as the number of input words.
    Unaligned tokens tend to join to the left aligned token.

    Args:
        matrix: a numpy array of shape (src_len, dst_len) filled with [0, 1, 2, 3], where 3 means a reliable alignment
         the corresponding words were aligned to one another in direct and reverse alignment runs, 1 and 2 mean that the
         words were aligned only in one direction, 0 - no alignment.
        dsttokens: e.g. ["_2", "0", "1", "4_"]
    Returns:
        targets: list of string tokens, with one-to-one correspondence to matrix.shape[0]

    Example:
        If we get
            matrix=[[3, 0, 0, 0]
                    [0, 2, 2, 3]]
            dsttokens=["_2", "0", "1", "4_"]
        it gives
            targets = ["_201", "4_"]
        Actually, this is a mistake instead of ["_20", "14_"]. That will be further corrected by regular expressions.
    """
    targets = []
    last_covered_dst_id = -1
    for i in range(len(matrix)):
        dstlist = []
        for j in range(last_covered_dst_id + 1, len(dsttokens)):
            # matrix[i][j] == 3: safe alignment point
            if matrix[i][j] == 3 or (
                j == last_covered_dst_id + 1
                and np.all(matrix[i, :] == 0)  # if the whole line does not have safe points
                and np.all(matrix[:, j] == 0)  # and the whole column does not have safe points, match them
            ):
                if len(targets) == 0:  # if this is first safe point, attach left unaligned columns to it, if any
                    for k in range(0, j):
                        if np.all(matrix[:, k] == 0):  # if column k does not have safe points
                            dstlist.append(dsttokens[k])
                        else:
                            break
                dstlist.append(dsttokens[j])
                last_covered_dst_id = j
                for k in range(j + 1, len(dsttokens)):
                    if np.all(matrix[:, k] == 0):  # if column k does not have safe points
                        dstlist.append(dsttokens[k])
                        last_covered_dst_id = k
                    else:
                        break

        if len(dstlist) > 0:
            targets.append("+".join(dstlist))
        else:
            targets.append("<DELETE>")
    return targets


def main() -> None:
    f = open(args.giza_dir + "/GIZA++." + args.giza_suffix, "r", encoding="utf-8")
    g = open(args.giza_dir + "/GIZA++reverse." + args.giza_suffix, "r", encoding="utf-8")
    out = open(args.giza_dir + "/" + args.out_filename, "w", encoding="utf-8")
    cache = {}
    good_count, not_mono_count, not_covered_count, exception_count = 0, 0, 0, 0
    n = 0
    while True:
        n += 3
        if n % 10000 == 0:
            print(n, "lines processed")
        fline1 = f.readline().strip()
        fline2 = f.readline().strip()
        fline3 = f.readline().strip()
        gline1 = g.readline().strip()
        gline2 = g.readline().strip()
        gline3 = g.readline().strip()
        if fline1 == "" and gline1 == "":
            break
        cache_key = fline1 + "\t" + fline2 + "\t" + gline1 + "\t" + gline2
        if cache_key in cache:
            out.write(cache[cache_key] + "\n")
            continue
        if fline1 == "" or gline1 == "" or fline2 == "" or gline2 == "" or fline3 == "" or gline3 == "":
            raise ValueError("Empty line: " + str(n))
        try:
            matrix, srctokens, dsttokens = fill_alignment_matrix(fline2, fline3, gline2, gline3)
        except Exception:
            print(fline1)
            print(fline2)
            print(fline3)
            print(gline1)
            print(gline2)
            print(gline3)
            exception_count += 1
            out_str = "-exception:\t" + fline2 + "\t" + gline2
            out.write(out_str + "\n")
            continue
        else:
            matrix[matrix <= 2] = 0  # leave only 1-to-1 alignment points
            if check_monotonicity(matrix):
                targets = get_targets(matrix, dsttokens)
                if len(targets) != len(srctokens):
                    raise ValueError(
                        "targets length doesn't match srctokens length: len(targets)="
                        + str(len(targets))
                        + "; len(srctokens)="
                        + str(len(srctokens))
                    )
                srcline = gline2
                dstline = fline2

                align = " ".join(targets)

                ban = False

                if align is None:
                    ban = True

                if ban:
                    out_str = (
                        "ban:\t"
                        + " ".join(srctokens)
                        + "\t"
                        + " ".join(dsttokens)
                        + "\t"
                        + str(align)
                    )
                else:
                    out_str = (
                        "good:\t"
                        + " ".join(srctokens)
                        + "\t"
                        + " ".join(dsttokens)
                        + "\t"
                        + align
                    )

                out.write(out_str + "\n")
                cache[cache_key] = out_str
            else:
                out_str = "-mon:\t" + " ".join(srctokens) + "\t" + " ".join(dsttokens)
                out.write(out_str + "\n")
                cache[cache_key] = out_str
                not_mono_count += 1

    f.close()
    g.close()
    out.close()


# Main code
if __name__ == '__main__':
    main()
