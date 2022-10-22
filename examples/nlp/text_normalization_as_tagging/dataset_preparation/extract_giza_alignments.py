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
parser.add_argument('--lang', type=str, required=True, help="Language")
args = parser.parse_args()

SHORT_ROMAN_NUMBERS = {
    "_i_",
    "_ii_",
    "_iii_",
    "_iiii_",
    "_iv_",
    "_v_",
    "_vi_",
    "_vii_",
    "_viii_",
    "_ix_",
    "_x_",
    "_xi_",
    "_xii_",
    "_xiii_",
    "_xiv_",
    "_xv_",
    "_xvi_",
    "_xvii_",
    "_xviii_",
    "_xix_",
    "_xx_",
    "_xxx_",
    "_xl_",
    "_l_",
    "_lx_",
    "_lxx_",
}


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
            targets.append("".join(dstlist))
        else:
            targets.append("<DELETE>")
    return targets


def get_targets_from_back(matrix: np.ndarray, dsttokens: List[str]) -> List[str]:
    """Join some of the destination tokens, so that their number becomes the same as the number of input words.
    Unaligned tokens tend to join to the right aligned token.

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
            targets = ["_2", "014_"]
        Actually, this is a mistake instead of ["_20", "14_"]. That will be further corrected by regular expressions.
    """

    targets = []
    last_covered_dst_id = len(dsttokens)
    for i in range(len(matrix) - 1, -1, -1):
        dstlist = []
        for j in range(last_covered_dst_id - 1, -1, -1):
            if matrix[i][j] == 3 or (
                j == last_covered_dst_id - 1 and np.all(matrix[i, :] == 0) and np.all(matrix[:, j] == 0)
            ):
                if len(targets) == 0:
                    for k in range(len(dsttokens) - 1, j, -1):
                        if np.all(matrix[:, k] == 0):
                            dstlist.append(dsttokens[k])
                        else:
                            break
                dstlist.append(dsttokens[j])
                last_covered_dst_id = j
                for k in range(j - 1, -1, -1):
                    if np.all(matrix[:, k] == 0):
                        dstlist.append(dsttokens[k])
                        last_covered_dst_id = k
                    else:
                        break
        if len(dstlist) > 0:
            targets.append("".join(list(reversed(dstlist))))
        else:
            targets.append("<DELETE>")
    return list(reversed(targets))


def main() -> None:
    g = open(args.giza_dir + "/GIZA++." + args.giza_suffix, "r", encoding="utf-8")
    f = open(args.giza_dir + "/GIZA++reverse." + args.giza_suffix, "r", encoding="utf-8")
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
                targets_from_back = get_targets_from_back(matrix, dsttokens)
                if len(targets) != len(srctokens):
                    raise ValueError(
                        "targets length doesn't match srctokens length: len(targets)="
                        + str(len(targets))
                        + "; len(srctokens)="
                        + str(len(srctokens))
                    )
                srcline = gline2

                leftside_align = " ".join(targets)
                rightside_align = " ".join(targets_from_back)

                rightside_align = rightside_align.replace("<DELETE> <DELETE> _11100_", "_11 <DELETE> 100_")
                leftside_align = leftside_align.replace("<DELETE> <DELETE> _11100_", "_11 <DELETE> 100_")

                # _1 4000_ => _14 000_
                # 1 5,000 => 15 ,000
                rightside_align = re.sub(r"^_1 ([\d],?_?)(_?000)", r"_1\g<1> \g<2>", rightside_align)
                leftside_align = re.sub(r"^_1 ([\d],?_?)(_?000)", r"_1\g<1> \g<2>", leftside_align)

                # "_2 10 0_" => "_2 <DELETE> 100_"
                rightside_align = re.sub(r"([\d]) 10 0_", r"\g<1> <DELETE> 100_", rightside_align)
                leftside_align = re.sub(r"([\d]) 10 0_", r"\g<1> <DELETE> 100_", leftside_align)

                if srctokens[0] in [
                    "ten",
                    "twenty",
                    "thirty",
                    "forty",
                    "fifty",
                    "sixty",
                    "seventy",
                    "eighty",
                    "ninety",
                ]:
                    #  ten thousand sixty  _1 00 60_  =>  _10 0 60_
                    rightside_align = re.sub(r"^(_\d) 00 (\d)", r"\g<1>0 0 \g<2>", rightside_align)
                    leftside_align = re.sub(r"^(_\d) 00 (\d)", r"\g<1>0 0 \g<2>", leftside_align)

                #  ten thousand sixty three    _1 0, 06 3_ => _10 ,0 6 3_
                rightside_align = re.sub(r"([ _]\d) 0, 0(\d)", r"\g<1>0 ,0 \g<2>", rightside_align)
                leftside_align = re.sub(r"([ _]\d) 0, 0(\d)", r"\g<1>0 ,0 \g<2>", leftside_align)

                # _4 0,__000_ => _40 ,__000_
                rightside_align = re.sub(r"(\d) 0(,__000_)", r"\g<1>0 \g<2>", rightside_align)
                leftside_align = re.sub(r"(\d) 0(,__000_)", r"\g<1>0 \g<2>", leftside_align)

                # _1 0 0,__000_ => _1 00 ,__000_
                rightside_align = re.sub(r"(\d) 0 0(,__000_)", r"\g<1> 00 \g<2>", rightside_align)
                leftside_align = re.sub(r"(\d) 0 0(,__000_)", r"\g<1> 00 \g<2>", leftside_align)

                #  7 <DELETE> 0__3 <DELETE> 9 2_ =>  70_ <DELETE> _3 <DELETE> 9 2_
                rightside_align = re.sub(r"(\d) <DELETE> (0_)(_\d)", r"\g<1>\g<2> <DELETE> \g<3>", rightside_align)
                leftside_align = re.sub(r"(\d) <DELETE> (0_)(_\d)", r"\g<1>\g<2> <DELETE> \g<3>", leftside_align)

                #  _3 0, 7 7 4=> _30 , 7 7 4_
                rightside_align = re.sub(r"(\d) 0, ", r"\g<1>0 , ", rightside_align)
                leftside_align = re.sub(r"(\d) 0, ", r"\g<1>0 , ", leftside_align)

                #   _1 1, 1 <DELETE> 40_  =>  _11 , 1 <DELETE> 40_
                rightside_align = re.sub(r"1 1, (\d)", r"11 , \g<1>", rightside_align)
                leftside_align = re.sub(r"1 1, (\d)", r"11 , \g<1>", leftside_align)

                #  <DELETE> 1 10_  =>  1 <DELETE> 10_  # two thousand one hundred ten
                if "one hundred" in srcline:
                    rightside_align = re.sub(r"<DELETE> 1 (1\d_)", r"1 <DELETE> \g<1>", rightside_align)
                    leftside_align = re.sub(r"<DELETE> 1 (1\d_)", r"1 <DELETE> \g<1>", leftside_align)

                # <DELETE> 10__hp_ => 10_ _hp_
                rightside_align = re.sub(r"<DELETE> (1\d_)_", r"\g<1> _", rightside_align)
                leftside_align = re.sub(r"<DELETE> (1\d_)_", r"\g<1> _", leftside_align)

                # _1 <DELETE> <DELETE> 2__200_ => _12_ <DELETE> _2 00_
                rightside_align = re.sub(
                    r"1 <DELETE> <DELETE> (\d_?)(_?\d)00_", r"1\g<1> <DELETE> \g<2> 00_", rightside_align
                )
                leftside_align = re.sub(
                    r"1 <DELETE> <DELETE> (\d_?)(_?\d)00_", r"1\g<1> <DELETE> \g<2> 00_", leftside_align
                )

                if re.match(r".+надцат", srctokens[0]) or srctokens[0] in [
                    "ten",
                    "eleven",
                    "twelve",
                    "thirteen",
                    "fourteen",
                    "fifteen",
                    "sixteen",
                    "seventeen",
                    "eighteen",
                    "nineteen",
                ]:
                    # "_1 <DELETE> 12 14_" -> "_11 <DELETE> 2 14_"
                    rightside_align = re.sub(
                        r"^(_1) (<DELETE>) ([\d]_?)(_?[\d])", r"\g<1>\g<3> \g<2> \g<4>", rightside_align
                    )
                    leftside_align = re.sub(
                        r"^(_1) (<DELETE>) ([\d]_?)(_?[\d])", r"\g<1>\g<3> \g<2> \g<4>", leftside_align
                    )

                    # "_1 10 10_" -> "_11 0 10_"
                    rightside_align = re.sub(
                        r"^_1 ([\d]_?)(_?0) ([\d] ?[\d])", r"_1\g<1> \g<2> \g<3>", rightside_align
                    )
                    leftside_align = re.sub(r"^_1 ([\d]_?)(_?0) ([\d] ?[\d])", r"_1\g<1> \g<2> \g<3>", leftside_align)

                if args.giza_dir.endswith("decimal") and args.lang == "ru":
                    # "_1 <DELETE> 0, 5_" => "_10 <DELETE> , 5_"      #десять целых и пять десятых
                    rightside_align = re.sub(
                        r"(\d) (<DELETE>) ([0123456789])(,) ([\d])", r"\g<1>\g<3> \g<2> \g<4> \g<5>", rightside_align
                    )
                    leftside_align = re.sub(
                        r"(\d) (<DELETE>) ([0123456789])(,) ([\d])", r"\g<1>\g<3> \g<2> \g<4> \g<5>", leftside_align
                    )

                if args.giza_dir.endswith("decimal") and args.lang == "en":
                    # "_7 0. 7_" => _70 . 7_
                    rightside_align = re.sub(r"^(_\d) 0\. (\d)", r"\g<1>0 . \g<2>", rightside_align)
                    leftside_align = re.sub(r"^(_\d) 0\. (\d)", r"\g<1>0 . \g<2>", leftside_align)

                if args.giza_dir.endswith("money") and args.lang == "en":
                    # "_1 , 000__£<<" => "_1 ,000_ _£<<"
                    rightside_align = re.sub(r"(\d) , 000_(_[£$€])", r"\g<1> ,000_ \g<2>", rightside_align)
                    leftside_align = re.sub(r"(\d) , 000_(_[£$€])", r"\g<1> ,000_ \g<2>", leftside_align)

                    # _18 <DELETE> .00__wst_   =>  _18.00_ <DELETE> _wst_
                    rightside_align = re.sub(r"(\d) <DELETE> \.00_(_)", r"\g<1>.00_ <DELETE> \g<2>", rightside_align)
                    leftside_align = re.sub(r"(\d) <DELETE> \.00_(_)", r"\g<1>.00_ <DELETE> \g<2>", leftside_align)

                    # _18 .00__£<<  =>   _18.00_ _£<<
                    rightside_align = re.sub(r"(\d) \.00_(_)", r"\g<1>.00_ \g<2>", rightside_align)
                    leftside_align = re.sub(r"(\d) \.00_(_)", r"\g<1>.00_ \g<2>", leftside_align)

                    #  8 6 , 000.00__$<<   =>  8 6 ,000.00_ _$<<
                    rightside_align = re.sub(r"(\d) , 000\.00_(_)", r"\g<1> ,000.00_ \g<2>", rightside_align)
                    leftside_align = re.sub(r"(\d) , 000\.00_(_)", r"\g<1> ,000.00_ \g<2>", leftside_align)

                    # 5 0 0.00__us__$<<  =>  5 00.00_ _us__$<<
                    rightside_align = re.sub(r"(\d) 0 0\.00_(_)", r"\g<1> 00.00_ \g<2>", rightside_align)
                    leftside_align = re.sub(r"(\d) 0 0\.00_(_)", r"\g<1> 00.00_ \g<2>", leftside_align)

                    # 8 0.00__$<<  => 80.00_ _$<<
                    rightside_align = re.sub(r"(\d) 0\.00_(_)", r"\g<1>0.00_ \g<2>", rightside_align)
                    leftside_align = re.sub(r"(\d) 0\.00_(_)", r"\g<1>0.00_ \g<2>", leftside_align)

                if args.giza_dir.endswith("money"):
                    # "_5 <DELETE> 000000__иен_" => "_5 000000_ _иен_"
                    rightside_align = re.sub(
                        r"([\d]_?) <DELETE> (_?000000_)(_[^\d])", r"\g<1> \g<2> \g<3>", rightside_align
                    )
                    leftside_align = re.sub(
                        r"([\d]_?) <DELETE> (_?000000_)(_[^\d])", r"\g<1> \g<2> \g<3>", leftside_align
                    )

                    # _5_ <DELETE> _m__£<< => "_5_ _m_ _£<<"
                    rightside_align = re.sub(
                        r"([\d]_) <DELETE> (_[mk]_)(_[^\d])", r"\g<1> \g<2> \g<3>", rightside_align
                    )
                    leftside_align = re.sub(r"([\d]_) <DELETE> (_[mk]_)(_[^\d])", r"\g<1> \g<2> \g<3>", leftside_align)

                    # "_3 <DELETE> 0__m__£<<" => "_30 _m_ _£<<"
                    rightside_align = re.sub(
                        r"([\d]) <DELETE> (0_)(_[mk]_)(_[^\d])", r"\g<1>\g<2> \g<3> \g<4>", rightside_align
                    )
                    leftside_align = re.sub(
                        r"([\d]) <DELETE> (0_)(_[mk]_)(_[^\d])", r"\g<1>\g<2> \g<3> \g<4>", leftside_align
                    )

                # "_15 <DELETE> 000__руб._" => "_15 000_ _руб._"
                rightside_align = re.sub(r"([\d]_?) <DELETE> (_?000_)(_[^\d])", r"\g<1> \g<2> \g<3>", rightside_align)
                leftside_align = re.sub(r"([\d]_?) <DELETE> (_?000_)(_[^\d])", r"\g<1> \g<2> \g<3>", leftside_align)

                # "_2 5 0 000__$<<" => "_2 50 000_ _$<<"
                rightside_align = re.sub(r"([\d]) (0_?) (_?000_)(_[^\d])", r"\g<1>\g<2> \g<3> \g<4>", rightside_align)
                leftside_align = re.sub(r"([\d]) (0_?) (_?000_)(_[^\d])", r"\g<1>\g<2> \g<3> \g<4>", leftside_align)

                # "_5 0 0000__$_" => "_500 000_ _$_"
                rightside_align = re.sub(
                    r"([\d]) 0 (0_?)(_?000_)(_[^\d])", r"\g<1>0\g<2> \g<3> \g<4>", rightside_align
                )
                leftside_align = re.sub(r"([\d]) 0 (0_?)(_?000_)(_[^\d])", r"\g<1>0\g<2> \g<3> \g<4>", leftside_align)

                # "_1 000__руб._" => "_1000_ _руб._"
                rightside_align = re.sub(r"(_1_?) (_?000_)(_[^\d])", r"\g<1>\g<2> \g<3>", rightside_align)
                leftside_align = re.sub(r"(_1_?) (_?000_)(_[^\d])", r"\g<1>\g<2> \g<3>", leftside_align)

                # replace cases like "2 0__января" with "20_ _января"
                leftside_align = re.sub(r"([\d]) (00?_)(_[^\d])", r"\g<1>\g<2> \g<3>", leftside_align)
                rightside_align = re.sub(r"([\d]) (00?_)(_[^\d])", r"\g<1>\g<2> \g<3>", rightside_align)

                #  "_3 <DELETE> 0__september_ _2 014_" => "_30_ <DELETE> _september_ _2 014_"
                #  "_3 <DELETE> 00__тыс.__руб._" => "_300_ <DELETE> _тыс.__руб._"
                leftside_align = re.sub(
                    r"([\d]) <DELETE> (00?_)(_[^\d])", r"\g<1>\g<2> <DELETE> \g<3>", leftside_align
                )
                rightside_align = re.sub(
                    r"([\d]) <DELETE> (00?_)(_[^\d])", r"\g<1>\g<2> <DELETE> \g<3>", rightside_align
                )

                # "_october_ _2 0,2 015_" => "_october_ _20 ,2 015_"
                leftside_align = re.sub(r"([\d]) (0),(\d)", r"\g<1>\g<2> ,\g<3>", leftside_align)
                rightside_align = re.sub(r"([\d]) (0),(\d)", r"\g<1>\g<2> ,\g<3>", rightside_align)

                # "_3 0_.10. _1 9 4 3_" =>  "_30_ .10. _1 9 4 3_"
                leftside_align = re.sub(r"([\d]) (0_)(\.[\d])", r"\g<1>\g<2> \g<3>", leftside_align)
                rightside_align = re.sub(r"([\d]) (0_)(\.[\d])", r"\g<1>\g<2> \g<3>", rightside_align)

                # _1 0000_ => _10 000_
                # _5 00000_ => _500 000_
                # _8 0__000_ =>  _80_ _000_
                rightside_align = re.sub(r"([\d]) (00?_?)(_?000000000_)", r"\g<1>\g<2> \g<3>", rightside_align)
                leftside_align = re.sub(r"([\d]) (00?_?)(_?000000000_)", r"\g<1>\g<2> \g<3>", leftside_align)
                rightside_align = re.sub(r"([\d]) (00?_?)(_?000000_)", r"\g<1>\g<2> \g<3>", rightside_align)
                leftside_align = re.sub(r"([\d]) (00?_?)(_?000000_)", r"\g<1>\g<2> \g<3>", leftside_align)
                rightside_align = re.sub(r"([\d]) (00?_?)(_?000_)", r"\g<1>\g<2> \g<3>", rightside_align)
                leftside_align = re.sub(r"([\d]) (00?_?)(_?000_)", r"\g<1>\g<2> \g<3>", leftside_align)

                # "_4 00,000_" -> "_400 ,000_"
                rightside_align = re.sub(r"([\d]) (00?),(000_)", r"\g<1>\g<2> ,\g<3>", rightside_align)
                leftside_align = re.sub(r"([\d]) (00?),(000_)", r"\g<1>\g<2> ,\g<3>", leftside_align)

                # "_4 <DELETE> 00,000_" -> "_4 00 ,000_"  # four hundred thousand
                # "_4 <DELETE> 00000_" -> "_4 00 000_"  # four hundred thousand
                rightside_align = re.sub(r"([\d]) <DELETE> (00)(,?000_)", r"\g<1> \g<2> \g<3>", rightside_align)
                leftside_align = re.sub(r"([\d]) <DELETE> (00)(,?000_)", r"\g<1> \g<2> \g<3>", leftside_align)

                # _6 <DELETE> 0002_ => _60 00 2_  # sixty thousand two
                # _2 <DELETE> 0002 000_ => _20 00 2 000_  # twenty million two thousand
                rightside_align = re.sub(
                    r"([\d]) <DELETE> (0)(00)([123456789])", r"\g<1>\g<2> \g<3> \g<4>", rightside_align
                )
                leftside_align = re.sub(
                    r"([\d]) <DELETE> (0)(00)([123456789])", r"\g<1>\g<2> \g<3> \g<4>", leftside_align
                )

                # _6 <DELETE> <DELETE> 00002_ => _6 00 00 2_ #six hundred thousand two
                rightside_align = re.sub(
                    r"([\d]) <DELETE> <DELETE> (00)(00)([123456789]_)", r"\g<1> \g<2> \g<3> \g<4>", rightside_align
                )
                leftside_align = re.sub(
                    r"([\d]) <DELETE> <DELETE> (00)(00)([123456789]_)", r"\g<1> \g<2> \g<3> \g<4>", leftside_align
                )

                # _2 <DELETE> 0012_ => _20 0 12_  # twenty thousand twelve
                rightside_align = re.sub(r"([\d]) <DELETE> (0)(0)(1\d)", r"\g<1>\g<2> \g<3> \g<4>", rightside_align)
                leftside_align = re.sub(r"([\d]) <DELETE> (0)(0)(1\d)", r"\g<1>\g<2> \g<3> \g<4>", leftside_align)

                # _1 <DELETE> 017_ => _1 0 17_ # one thousand seventeen
                rightside_align = re.sub(r"([\d]) <DELETE> (0)(1\d)", r"\g<1> \g<2> \g<3>", rightside_align)
                leftside_align = re.sub(r"([\d]) <DELETE> (0)(1\d)", r"\g<1> \g<2> \g<3>", leftside_align)

                # "_9 3 ,0__²_> _км_" => "_9 3 ,0__²_> _км_"
                rightside_align = re.sub(r"([\d]) (,00?_)(_[^\d])", r"\g<1>\g<2> \g<3>", rightside_align)
                leftside_align = re.sub(r"([\d]) (,00?_)(_[^\d])", r"\g<1>\g<2> \g<3>", leftside_align)

                # "_0 <DELETE> , <DELETE> <DELETE> 01__г_" => "_0 <DELETE> , 01 <DELETE> _г_"
                rightside_align = re.sub(
                    r"(,) <DELETE> <DELETE> 01_(_[^\d])", r"\g<1> 01_ <DELETE> \g<2>", rightside_align
                )
                leftside_align = re.sub(
                    r"(,) <DELETE> <DELETE> 01_(_[^\d])", r"\g<1> 01_ <DELETE> \g<2>", leftside_align
                )

                # "_0 <DELETE> , 7 6 <DELETE> <DELETE> 1__км_" => "_0 <DELETE> , 7 6 1_ <DELETE> _км_"
                rightside_align = re.sub(
                    r"(,) (\d) (\d) <DELETE> <DELETE> 1_(_[^\d])",
                    r"\g<1> \g<2> \g<3> 1_ <DELETE> \g<4>",
                    rightside_align,
                )
                leftside_align = re.sub(
                    r"(,) (\d) (\d) <DELETE> <DELETE> 1_(_[^\d])",
                    r"\g<1> \g<2> \g<3> 1_ <DELETE> \g<4>",
                    leftside_align,
                )

                # "_5 <DELETE> 0000__рублей_" => "_50 000_ рублей"
                rightside_align = re.sub(
                    r"([\d]) <DELETE> ([0][0]?_?)(_?000_)(_)", r"\g<1>\g<2> \g<3> \g<4>", rightside_align
                )
                leftside_align = re.sub(
                    r"([\d]) <DELETE> ([0][0]?_?)(_?000_)(_)", r"\g<1>\g<2> \g<3> \g<4>", leftside_align
                )

                # _3 , 000__m_  => _3 ,000_ _m_
                # _3 , 000__lb_  => _3 ,000_ _lb_
                rightside_align = re.sub(r"(\d) , 000_(_)", r"\g<1> ,000_ \g<2>", rightside_align)
                leftside_align = re.sub(r"(\d) , 000_(_)", r"\g<1> ,000_ \g<2>", leftside_align)

                # 50 , <DELETE> 000__sar_  => 50 ,000_ _sar_
                rightside_align = re.sub(r"(\d) , <DELETE> 000_(_)", r"\g<1> ,000_ <DELETE> \g<2>", rightside_align)
                leftside_align = re.sub(r"(\d) , <DELETE> 000_(_)", r"\g<1> ,000_ <DELETE> \g<2>", leftside_align)

                # _2 , 000,000__$<<  => _2 ,000,000_ _$<<
                rightside_align = re.sub(r"(\d) , 000,000_(_)", r"\g<1> ,000,000_ \g<2>", rightside_align)
                leftside_align = re.sub(r"(\d) , 000,000_(_)", r"\g<1> ,000,000_ \g<2>", leftside_align)

                #  _40 ,000 ,000__$<<  =>  _40 ,000,000_ _$<<
                rightside_align = re.sub(r"(\d) ,000 ,000_(_)", r"\g<1> ,000,000_ \g<2>", rightside_align)
                leftside_align = re.sub(r"(\d) ,000 ,000_(_)", r"\g<1> ,000,000_ \g<2>", leftside_align)

                # "_1 <DELETE> 115_" -> "_1 1 15_"
                rightside_align = re.sub(r"<DELETE> ([1])([1][\d])", r"\g<1> \g<2>", rightside_align)
                leftside_align = re.sub(r"<DELETE> ([1])([1][\d])", r"\g<1> \g<2>", leftside_align)

                # "_1 <DELETE> 990-х_" -> "_1 9 90-х_"
                rightside_align = re.sub(r"<DELETE> (9)(90)", r"\g<1> \g<2>", rightside_align)
                leftside_align = re.sub(r"<DELETE> (9)(90)", r"\g<1> \g<2>", leftside_align)
                rightside_align = re.sub(r"<DELETE> (8)(80)", r"\g<1> \g<2>", rightside_align)
                leftside_align = re.sub(r"<DELETE> (8)(80)", r"\g<1> \g<2>", leftside_align)
                rightside_align = re.sub(r"<DELETE> (7)(70)", r"\g<1> \g<2>", rightside_align)
                leftside_align = re.sub(r"<DELETE> (7)(70)", r"\g<1> \g<2>", leftside_align)
                rightside_align = re.sub(r"<DELETE> (6)(60)", r"\g<1> \g<2>", rightside_align)
                leftside_align = re.sub(r"<DELETE> (6)(60)", r"\g<1> \g<2>", leftside_align)
                rightside_align = re.sub(r"<DELETE> (5)(50)", r"\g<1> \g<2>", rightside_align)
                leftside_align = re.sub(r"<DELETE> (5)(50)", r"\g<1> \g<2>", leftside_align)
                rightside_align = re.sub(r"<DELETE> (4)(40)", r"\g<1> \g<2>", rightside_align)
                leftside_align = re.sub(r"<DELETE> (4)(40)", r"\g<1> \g<2>", leftside_align)
                rightside_align = re.sub(r"<DELETE> (3)(30)", r"\g<1> \g<2>", rightside_align)
                leftside_align = re.sub(r"<DELETE> (3)(30)", r"\g<1> \g<2>", leftside_align)
                rightside_align = re.sub(r"<DELETE> (2)(20)", r"\g<1> \g<2>", rightside_align)
                leftside_align = re.sub(r"<DELETE> (2)(20)", r"\g<1> \g<2>", leftside_align)

                # восемь ноль ноль ноль ноль ноль ноль ноль _8 0 0 0 0 0 0 0_
                # _8 <DELETE> <DELETE> <DELETE> <DELETE> <DELETE> <DELETE> 0000000_
                rightside_align = re.sub(
                    r"<DELETE> <DELETE> <DELETE> <DELETE> <DELETE> <DELETE> 0000000_",
                    r"0 0 0 0 0 0 0_",
                    rightside_align,
                )
                leftside_align = re.sub(
                    r"<DELETE> <DELETE> <DELETE> <DELETE> <DELETE> <DELETE> 0000000_",
                    r"0 0 0 0 0 0 0_",
                    leftside_align,
                )

                # _8 <DELETE> <DELETE> <DELETE> <DELETE> <DELETE> 000000_
                rightside_align = re.sub(
                    r"<DELETE> <DELETE> <DELETE> <DELETE> <DELETE> 000000_", r"0 0 0 0 0 0_", rightside_align
                )
                leftside_align = re.sub(
                    r"<DELETE> <DELETE> <DELETE> <DELETE> <DELETE> 000000_", r"0 0 0 0 0 0_", leftside_align
                )

                # _8 <DELETE> <DELETE> <DELETE> <DELETE> 00000_
                rightside_align = re.sub(r"<DELETE> <DELETE> <DELETE> <DELETE> 00000_", r"0 0 0 0 0_", rightside_align)
                leftside_align = re.sub(r"<DELETE> <DELETE> <DELETE> <DELETE> 00000_", r"0 0 0 0 0_", leftside_align)

                # _8 <DELETE> <DELETE> <DELETE> 0000_
                rightside_align = re.sub(r"<DELETE> <DELETE> <DELETE> 0000_", r"0 0 0 0_", rightside_align)
                leftside_align = re.sub(r"<DELETE> <DELETE> <DELETE> 0000_", r"0 0 0 0_", leftside_align)

                # _8 <DELETE> <DELETE> 000_
                rightside_align = re.sub(r"<DELETE> <DELETE> 000_", r"0 0 0_", rightside_align)
                leftside_align = re.sub(r"<DELETE> <DELETE> 000_", r"0 0 0_", leftside_align)

                # "_2 <DELETE> <DELETE> 010/11" => "_2 0 10 /11"
                rightside_align = re.sub(
                    r"<DELETE> <DELETE> (0)([1][\d])/([\d])", r"\g<1> \g<2> /\g<3>", rightside_align
                )
                leftside_align = re.sub(
                    r"<DELETE> <DELETE> (0)([1][\d])/([\d])", r"\g<1> \g<2> /\g<3>", leftside_align
                )

                # "_2 0 <DELETE> 11/12_" => "_2 0 11 /12_"
                rightside_align = re.sub(r"<DELETE> ([\d]+)/([\d])", r"\g<1> /\g<2>", rightside_align)
                leftside_align = re.sub(r"<DELETE> ([\d]+)/([\d])", r"\g<1> /\g<2>", leftside_align)

                # "_2 0 1 0/2 0 11_" => "_2 0 10 /2 0 11_"
                rightside_align = re.sub(r"([\d]) ([\d]+)/([\d])", r"\g<1>\g<2> /\g<3>", rightside_align)
                leftside_align = re.sub(r"([\d]) ([\d]+)/([\d])", r"\g<1>\g<2> /\g<3>", leftside_align)

                # "_5 0%_" => "_50 %_"
                # "_1 00%_" => "_100 %_"
                # "_1 00,00%_" => "_100,00 %_"
                rightside_align = re.sub(r"([\d]) ([0,]+)%", r"\g<1>\g<2> %", rightside_align)
                leftside_align = re.sub(r"([\d]) ([0,]+)%", r"\g<1>\g<2> %", leftside_align)

                if args.lang == "en" and " and a " in srcline:
                    # _3 <DELETE> <DELETE> 0__1/2_ => _30_ <DELETE> <DELETE> _1/2_
                    rightside_align = re.sub(
                        r"(\d) <DELETE> <DELETE> 0__1(/\d_)", r"\g<1>0_ <DELETE> _1 \g<2>", rightside_align
                    )
                    leftside_align = re.sub(
                        r"(\d) <DELETE> <DELETE> 0__1(/\d_)", r"\g<1>0_ <DELETE> _1 \g<2>", leftside_align
                    )

                    # <DELETE> <DELETE> _1/2_ => <DELETE> _1 /2_
                    rightside_align = re.sub(r"<DELETE> <DELETE> _1(/\d_)", r"<DELETE> _1 \g<1>", rightside_align)
                    leftside_align = re.sub(r"<DELETE> <DELETE> _1(/\d_)", r"<DELETE> _1 \g<1>", leftside_align)

                # "_1 50_ <DELETE> _тыс.__руб._"  => "_1 50_ _тыс._ _руб._"
                rightside_align = re.sub(r"_ <DELETE> (_[^\d]+_)(_[^\d]+_)", r"_ \g<1> \g<2>", rightside_align)
                leftside_align = re.sub(r"_ <DELETE> (_[^\d]+_)(_[^\d]+_)", r"_ \g<1> \g<2>", leftside_align)

                # _1000 000__$_ => "_1000000_ _$_"
                rightside_align = re.sub(r"(_1_?_?000_?) (_?000_)(_[^\d])", r"\g<1>\g<2> \g<3>", rightside_align)
                leftside_align = re.sub(r"(_1_?_?000_?) (_?000_)(_[^\d])", r"\g<1>\g<2> \g<3>", leftside_align)

                if args.giza_dir.endswith("ordinal") and args.lang == "en":
                    # _5 <DELETE> 00,000th_  =>  _5 00 ,000th_ #five hundred thousandth
                    leftside_align = re.sub(r"(\d) <DELETE> 00,000th", r"\g<1> 00 ,000th", leftside_align)
                    rightside_align = re.sub(r"(\d) <DELETE> 00,000th", r"\g<1> 00 ,000th", rightside_align)

                if args.giza_dir.endswith("fraction") and args.lang == "en":
                    # <DELETE> _10/15_  => _10 /15_
                    leftside_align = re.sub(r"<DELETE> _10(/1\d)", r"_10 \g<1>", leftside_align)
                    rightside_align = re.sub(r"<DELETE> _10(/1\d)", r"_10 \g<1>", rightside_align)

                    # <DELETE> _1/12_ =>  _1 /12_    #one twelth
                    leftside_align = re.sub(r"<DELETE> _1(/1\d)", r"_1 \g<1>", leftside_align)
                    rightside_align = re.sub(r"<DELETE> _1(/1\d)", r"_1 \g<1>", rightside_align)

                    # _5 0/ 1_ => _50 / 1_  # fifty over one
                    leftside_align = re.sub(r"(\d) 0/ (\d)", r"\g<1>0 / \g<2>", leftside_align)
                    rightside_align = re.sub(r"(\d) 0/ (\d)", r"\g<1>0 / \g<2>", rightside_align)

                if args.giza_dir.endswith("date") and args.lang == "en":
                    #  <DELETE> <DELETE> _1222_ => _12 2 2_    # twelve twenty two
                    leftside_align = re.sub(r"<DELETE> <DELETE> _1222_", r"_12 2 2_", leftside_align)
                    rightside_align = re.sub(r"<DELETE> <DELETE> _1222_", r"_12 2 2_", rightside_align)

                    #  "_1 2_ <DELETE> _november_ _2 014_" => " <DELETE> _12_ <DELETE> _november_ _2 014_"
                    # _,__1 9_ <DELETE> => <DELETE> _,__19_ <DELETE>
                    if "the " in srcline:
                        leftside_align = re.sub(
                            r"^([_,.]*_1) (\d_) <DELETE>", r"<DELETE> \g<1>\g<2> <DELETE>", leftside_align
                        )
                        rightside_align = re.sub(
                            r"^([_,.]*_1) (\d_) <DELETE>", r"<DELETE> \g<1>\g<2> <DELETE>", rightside_align
                        )

                        leftside_align = re.sub(
                            r" ([_,.]*_1) (\d_) <DELETE>", r" <DELETE> \g<1>\g<2> <DELETE>", leftside_align
                        )
                        rightside_align = re.sub(
                            r" ([_,.]*_1) (\d_) <DELETE>", r" <DELETE> \g<1>\g<2> <DELETE>", rightside_align
                        )

                        leftside_align = re.sub(
                            r"^([_,.]*_1) (\dth_) <DELETE>", r"<DELETE> \g<1>\g<2> <DELETE>", leftside_align
                        )
                        rightside_align = re.sub(
                            r"^([_,.]*_1) (\dth_) <DELETE>", r"<DELETE> \g<1>\g<2> <DELETE>", rightside_align
                        )

                        leftside_align = re.sub(
                            r" ([_,.]*_1) (\dth_) <DELETE>", r" <DELETE> \g<1>\g<2> <DELETE>", leftside_align
                        )
                        rightside_align = re.sub(
                            r" ([_,.]*_1) (\dth_) <DELETE>", r" <DELETE> \g<1>\g<2> <DELETE>", rightside_align
                        )

                    # <DELETE> <DELETE> _12,2012_ => _12_ ,20 12_
                    leftside_align = re.sub(
                        r"^<DELETE> <DELETE> (_12)(,_?_?20)(12_)", r"\g<1> \g<2> \g<3>", leftside_align
                    )
                    rightside_align = re.sub(
                        r"^<DELETE> <DELETE> (_12)(,_?_?20)(12_)", r"\g<1> \g<2> \g<3>", rightside_align
                    )

                    #  <DELETE> <DELETE> .12.2012_ => .12. 20 12_
                    leftside_align = re.sub(
                        r"<DELETE> <DELETE> (\.12\.)(20)(1\d_)", r"\g<1> \g<2> \g<3>", leftside_align
                    )
                    rightside_align = re.sub(
                        r"<DELETE> <DELETE> (\.12\.)(20)(1\d_)", r"\g<1> \g<2> \g<3>", rightside_align
                    )

                    # <DELETE> <DELETE> .12.200 6_ => .12. 2 00 6_
                    leftside_align = re.sub(
                        r"<DELETE> <DELETE> (\.12\.)(2)(00)([1-9]_)", r"\g<1> \g<2> \g<3> \g<4>", leftside_align
                    )
                    rightside_align = re.sub(
                        r"<DELETE> <DELETE> (\.12\.)(2)(00)([1-9]_)", r"\g<1> \g<2> \g<3> \g<4>", rightside_align
                    )

                    # <DELETE> <DELETE> _1,__1919_ => _1 ,__19 19_
                    leftside_align = re.sub(
                        r"^<DELETE> <DELETE> (_1)(,_?_?19)(19)", r"\g<1> \g<2> \g<3>", leftside_align
                    )
                    rightside_align = re.sub(
                        r"^<DELETE> <DELETE> (_1)(,_?_?19)(19)", r"\g<1> \g<2> \g<3>", rightside_align
                    )

                    #  <DELETE> <DELETE> _19,__199 9_ => _19 ,__19 9 9_
                    leftside_align = re.sub(
                        r"^<DELETE> <DELETE> (_19)(,_?_?19)(9 )", r"\g<1> \g<2> \g<3>", leftside_align
                    )
                    rightside_align = re.sub(
                        r"^<DELETE> <DELETE> (_19)(,_?_?19)(9 )", r"\g<1> \g<2> \g<3>", rightside_align
                    )

                    leftside_align = re.sub(
                        r"^<DELETE> <DELETE> (_18)(,_?_?18)(8 )", r"\g<1> \g<2> \g<3>", leftside_align
                    )
                    rightside_align = re.sub(
                        r"^<DELETE> <DELETE> (_18)(,_?_?18)(8 )", r"\g<1> \g<2> \g<3>", rightside_align
                    )

                    leftside_align = re.sub(
                        r"^<DELETE> <DELETE> (_17)(,_?_?17)(7 )", r"\g<1> \g<2> \g<3>", leftside_align
                    )
                    rightside_align = re.sub(
                        r"^<DELETE> <DELETE> (_17)(,_?_?17)(7 )", r"\g<1> \g<2> \g<3>", rightside_align
                    )

                    leftside_align = re.sub(
                        r"^<DELETE> <DELETE> (_16)(,_?_?16)(6 )", r"\g<1> \g<2> \g<3>", leftside_align
                    )
                    rightside_align = re.sub(
                        r"^<DELETE> <DELETE> (_16)(,_?_?16)(6 )", r"\g<1> \g<2> \g<3>", rightside_align
                    )

                    #  <DELETE> <DELETE> _19,__1990_ => _19 ,__19 90_
                    leftside_align = re.sub(
                        r"^<DELETE> <DELETE> (_19)(,_?_?19)(90_)", r"\g<1> \g<2> \g<3>", leftside_align
                    )
                    rightside_align = re.sub(
                        r"^<DELETE> <DELETE> (_19)(,_?_?19)(90_)", r"\g<1> \g<2> \g<3>", rightside_align
                    )

                    leftside_align = re.sub(
                        r"^<DELETE> <DELETE> (_18)(,_?_?18)(80_)", r"\g<1> \g<2> \g<3>", leftside_align
                    )
                    rightside_align = re.sub(
                        r"^<DELETE> <DELETE> (_18)(,_?_?18)(80_)", r"\g<1> \g<2> \g<3>", rightside_align
                    )

                    # <DELETE> _1,__19 9 3_ => _1 ,__19 9 3_
                    # <DELETE> _1,__20 13_ => _1 ,__20 13_
                    # <DELETE> _1,__2 0 13_ => _1 ,__2 0 13_
                    leftside_align = re.sub(r"<DELETE> (_0?\d)(,_?_?[12])", r"\g<1> \g<2>", leftside_align)
                    rightside_align = re.sub(r"<DELETE> (_0?\d)(,_?_?[12])", r"\g<1> \g<2>", rightside_align)

                    #  <DELETE> _1,__20 14_
                    # "<DELETE> _1,20 14_" => "_1 ,20 14_"
                    # <DELETE> _2,__20 14_  => "_2 ,20 14_"
                    leftside_align = re.sub(r"^<DELETE> (_\d)(,_?_?\d)", r"\g<1> \g<2>", leftside_align)
                    rightside_align = re.sub(r"^<DELETE> (_\d)(,_?_?\d)", r"\g<1> \g<2>", rightside_align)

                    # "_2 <DELETE> 1,20 14_" => "_2 1 ,20 14_"
                    leftside_align = re.sub(r"<DELETE> 1,(_?_?\d)", r"1 ,\g<1>", leftside_align)
                    rightside_align = re.sub(r"<DELETE> 1,(_?_?\d)", r"1 ,\g<1>", rightside_align)

                    #  <DELETE> _11,19 9 7_  =>   _11 ,19 9 7_
                    leftside_align = re.sub(r"<DELETE> _11,(_?_?\d)", r"_11 ,\g<1>", leftside_align)
                    rightside_align = re.sub(r"<DELETE> _11,(_?_?\d)", r"_11 ,\g<1>", rightside_align)

                    if len(srctokens) >= 2 and srctokens[-2] == "twenty":
                        # "<DELETE> <DELETE> _12,200 9_" => "_12 ,20 09_"
                        leftside_align = re.sub(
                            r"<DELETE> <DELETE> _12(,_?_?20)0 (\d_)", r"_12 \g<1> 0\g<2>", leftside_align
                        )
                        rightside_align = re.sub(
                            r"<DELETE> <DELETE> _12(,_?_?20)0 (\d_)", r"_12 \g<1> 0\g<2>", rightside_align
                        )

                        # <DELETE> <DELETE> _10,__2013_ => _10 ,__20 13_ # january tenth twenty thirteen
                        leftside_align = re.sub(
                            r"<DELETE> <DELETE> (_1\d)(,_?_?20)(1\d_)", r"\g<1> \g<2> \g<3>", leftside_align
                        )
                        rightside_align = re.sub(
                            r"<DELETE> <DELETE> (_1\d)(,_?_?20)(1\d_)", r"\g<1> \g<2> \g<3>", rightside_align
                        )

                        # "_april_ _2 015_" => "_april_ _20 15_"
                        leftside_align = re.sub(r"2 0(\d\d_)$", r"20 \g<1>", leftside_align)
                        rightside_align = re.sub(r"2 0(\d\d_)$", r"20 \g<1>", rightside_align)

                        # _2 010s_ => _20 10s_    # twenty tens
                        # _2 010's_ => _20 10's_
                        leftside_align = re.sub(r"2 0(10'?s_)", r"20 \g<1>", leftside_align)
                        rightside_align = re.sub(r"2 0(10'?s_)", r"20 \g<1>", rightside_align)

                    elif len(srctokens) >= 2 and srctokens[-2] == "thousand":
                        # "<DELETE> <DELETE> _12,200 9_" => "_12 ,2 00 9_"
                        leftside_align = re.sub(
                            r"^<DELETE> <DELETE> _12(,_?_?2)00 (\d_)", r"_12 \g<1> 00 \g<2>", leftside_align
                        )
                        rightside_align = re.sub(
                            r"^<DELETE> <DELETE> _12(,_?_?2)00 (\d_)", r"_12 \g<1> 00 \g<2>", rightside_align
                        )

                    # thirtieth twenty fifteen   _3 0th__,20 15_ => _30th_ _,20 15_
                    leftside_align = re.sub(r"(\d) 0th_(_,?\d)", r"\g<1>0th_ \g<2>", leftside_align)
                    rightside_align = re.sub(r"(\d) 0th_(_,?\d)", r"\g<1>0th_ \g<2>", rightside_align)

                    # _2 0,__20 15_ => _20 ,__20 15
                    leftside_align = re.sub(r"([123]) 0(,_?_?\d)", r"\g<1>0 \g<2>", leftside_align)
                    rightside_align = re.sub(r"([123]) 0(,_?_?\d)", r"\g<1>0 \g<2>", rightside_align)

                if args.giza_dir.endswith("date") and args.lang == "ru":
                    # тысяча девятьсот шестидесятого года  _1 9 6 0_  => _1 9 60_ <DELETE>
                    if srctokens[-1] == "года":
                        leftside_align = re.sub(r"(\d) 0_", r"\g<1>0_ <DELETE>", leftside_align)
                        rightside_align = re.sub(r"(\d) 0_", r"\g<1>0_ <DELETE>", rightside_align)

                if args.giza_dir.endswith("time"):
                    if srctokens[-1] == "hundred":
                        # fifteen hundred     <DELETE> _15:00_
                        rightside_align = re.sub(r"<DELETE> (_\d\d:)00_", r"\g<1> 00_", rightside_align)
                        leftside_align = re.sub(r"<DELETE> (_\d\d:)00_", r"\g<1> 00_", leftside_align)

                        #  !! Do not change the order of next two rules
                        # twenty one hundred      _2 1:00_ <DELETE>
                        rightside_align = re.sub(r"(_\d) (\d:)00_ <DELETE>", r"\g<1> \g<2> 00_", rightside_align)
                        leftside_align = re.sub(r"(_\d) (\d:)00_ <DELETE>", r"\g<1> \g<2> 00_", leftside_align)
                        # twenty hundred      _2 0:00_
                        rightside_align = re.sub(r"(_\d) (\d:)00_", r"\g<1>\g<2> 00_", rightside_align)
                        leftside_align = re.sub(r"(_\d) (\d:)00_", r"\g<1>\g<2> 00_", leftside_align)

                    if srctokens[1] == "o'clock":
                        #  nine o'clock    <DELETE> _09:00_   => "_09:00_ <DELETE>"
                        rightside_align = re.sub(r"^<DELETE> ([^ ]+)", r"\g<1> <DELETE>", rightside_align)
                        leftside_align = re.sub(r"^<DELETE> ([^ ]+)", r"\g<1> <DELETE>", leftside_align)

                    if srctokens[0] == "one":
                        # <DELETE> _1:__2 8_  =>  _1:_ _2 8_   # one twenty eight
                        rightside_align = re.sub(r"^<DELETE> (_1:_?)(_?\d)", r"\g<1> \g<2>", rightside_align)
                        leftside_align = re.sub(r"^<DELETE> (_1:_?)(_?\d)", r"\g<1> \g<2>", leftside_align)

                    # _1 1:3 3_ => _11: 3 3_
                    # _1 1:__2 9_ => _11:_ _2 9_
                    rightside_align = re.sub(r"(_\d) (\d:_?)(_?\d)", r"\g<1>\g<2> \g<3>", rightside_align)
                    leftside_align = re.sub(r"(_\d) (\d:_?)(_?\d)", r"\g<1>\g<2> \g<3>", leftside_align)

                    # _3 :__11_  => _3:_ _11_
                    rightside_align = re.sub(r"(\d) (:_?)(_?\d)", r"\g<1>\g<2> \g<3>", rightside_align)
                    leftside_align = re.sub(r"(\d) (:_?)(_?\d)", r"\g<1>\g<2> \g<3>", leftside_align)

                if args.giza_dir.endswith("measure"):
                    # _6 <DELETE> <DELETE> 0__mph_ => _60_ <DELETE> <DELETE> _mph_
                    rightside_align = re.sub(
                        r"(\d) <DELETE> <DELETE> 0__(mph|kmh|rpm|kwh|gwh|mwh|twh|kph|mmhg|bpd)",
                        r"\g<1>0_ <DELETE> <DELETE> _\g<2>",
                        rightside_align,
                    )
                    leftside_align = re.sub(
                        r"(\d) <DELETE> <DELETE> 0__(mph|kmh|rpm|kwh|gwh|mwh|twh|kph|mmhg|bpd)",
                        r"\g<1>0_ <DELETE> <DELETE> _\g<2>",
                        leftside_align,
                    )

                    # _6 0 <DELETE> <DELETE> 00__rpm_  => _6 000_ <DELETE> <DELETE> _rpm_   # six thousand revolutions per minute
                    rightside_align = re.sub(
                        r"0 <DELETE> <DELETE> 00__(mph|kmh|rpm|kwh|gwh|mwh|twh|kph|mmhg|bpd)",
                        r"000_ <DELETE> <DELETE> _\g<1>",
                        rightside_align,
                    )
                    leftside_align = re.sub(
                        r"0 <DELETE> <DELETE> 00__(mph|kmh|rpm|kwh|gwh|mwh|twh|kph|mmhg|bpd)",
                        r"000_ <DELETE> <DELETE> _\g<1>",
                        leftside_align,
                    )

                    # _6 , 0 0 0__rpm_  #six thousand revolutions per minute
                    rightside_align = re.sub(
                        r"(\d) , 0 0 0__(mph|kmh|rpm|kwh|gwh|mwh|twh|kph|mmhg|bpd)",
                        r"\g<1> ,000_ <DELETE> <DELETE> _\g<2>",
                        rightside_align,
                    )
                    leftside_align = re.sub(
                        r"(\d) , 0 0 0__(mph|kmh|rpm|kwh|gwh|mwh|twh|kph|mmhg|bpd)",
                        r"\g<1> ,000_ <DELETE> <DELETE> _\g<2>",
                        leftside_align,
                    )

                    # _2 0"_ => _20 "_    # twenty inches
                    rightside_align = re.sub(r"(\d) 0([\"'])", r"\g<1>0 \g<2>", rightside_align)
                    leftside_align = re.sub(r"(\d) 0([\"'])", r"\g<1>0 \g<2>", leftside_align)

                ban = False

                # ban multiword roman numbers, because we do not split them to parts
                for t in rightside_align.split():
                    if t not in SHORT_ROMAN_NUMBERS and re.match(r"^_?[civxl][civxl]+_?$", t):
                        ban = True

                # ban cases like "_11/05/2013_", "_2005-11-25_", because they are source of incorrect alignments
                if args.giza_dir.endswith("date") and args.lang == "en":
                    if "/" in rightside_align or "-" in rightside_align:
                        ban = True

                if args.giza_dir.endswith("ordinal"):
                    if "º" in rightside_align or "ª" in rightside_align:
                        ban = True

                if args.giza_dir.endswith("cardinal"):
                    # ban cases like  "two hundred sixty" => "_2 60__a_" because they are corpus mistakes
                    if re.match(r".+\d__\w", rightside_align):
                        ban = True
                    if "_u.__s._" in rightside_align:
                        ban = True
                    if ":" in rightside_align or "-_" in rightside_align:
                        ban = True

                if args.giza_dir.endswith("money"):
                    if (
                        rightside_align.startswith("_a__")
                        or rightside_align.startswith("_s__")
                        or rightside_align.startswith("_hk__")
                        or rightside_align.startswith("_rs__")
                        or rightside_align.startswith("_nz__")
                        or rightside_align.startswith("_nt__")
                    ):
                        ban = True

                # ban brackets
                if "(" in rightside_align or ")" in rightside_align:
                    ban = True

                if ban:
                    out_str = (
                        "ban:\t"
                        + " ".join(srctokens)
                        + "\t"
                        + " ".join(dsttokens)
                        + "\t"
                        + leftside_align
                        + "\t"
                        + rightside_align
                    )
                else:
                    out_str = (
                        "good:\t"
                        + " ".join(srctokens)
                        + "\t"
                        + " ".join(dsttokens)
                        + "\t"
                        + leftside_align
                        + "\t"
                        + rightside_align
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
