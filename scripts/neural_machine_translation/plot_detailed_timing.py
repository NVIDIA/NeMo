#!/usr/bin/env python3
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
This script takes as an input XXXX.json files
(i.e., the output of nmt_transformer_infer.py --write_timing)
and creates plots XXX.PLOT_NAME.png at the same path.
"""
import json
import os
import sys

from matplotlib import pyplot as plt

# =============================================================================#
# Control Variables
# =============================================================================#

PLOTS_EXT = "pdf"
PLOT_TITLE = False
PLOT_XLABEL = True
PLOT_YLABEL = True
PLOT_LABEL_FONT_SIZE = 16
PLOT_GRID = True

# =============================================================================#
# Helper functions
# =============================================================================#


def plot_timing(lengths, timings, lengths_name, timings_name, fig=None):
    if fig is None:
        fig = plt.figure()

    plt.scatter(lengths, timings, label=timings_name)
    if PLOT_XLABEL:
        plt.xlabel(f"{lengths_name} [tokens]", fontsize=PLOT_LABEL_FONT_SIZE)
    if PLOT_YLABEL:
        plt.ylabel(f"{timings_name} [sec]", fontsize=PLOT_LABEL_FONT_SIZE)
    if PLOT_GRID:
        plt.grid(True)
    if PLOT_TITLE:
        plt.title(f"{timings_name} vs. {lengths_name}")

    plt.xticks(fontsize=PLOT_LABEL_FONT_SIZE)
    plt.yticks(fontsize=PLOT_LABEL_FONT_SIZE)
    plt.tight_layout()

    return fig


# =============================================================================#
# Main script
# =============================================================================#
if __name__ == "__main__":
    print("Usage: plot_detailed_timing.py <JSON FILE> <SJON FILE> ...")
    for timing_fn in sys.argv[1:]:
        # load data
        print(f"Parsing file = {timing_fn}")
        data = json.load(open(timing_fn))

        # plot data
        gifs_dict = {}
        gifs_dict["encoder-src_len"] = plot_timing(
            lengths=data["mean_src_length"],
            timings=data["encoder"],
            lengths_name="src length",
            timings_name="encoder",
        )
        gifs_dict["sampler-src_len"] = plot_timing(
            lengths=data["mean_src_length"],
            timings=data["sampler"],
            lengths_name="src length",
            timings_name="sampler",
        )
        gifs_dict["sampler-tgt_len"] = plot_timing(
            lengths=data["mean_tgt_length"],
            timings=data["sampler"],
            lengths_name="tgt length",
            timings_name="sampler",
        )

        # save data
        base_fn = os.path.splitext(timing_fn)[0]
        for name, fig in gifs_dict.items():
            plot_fn = f"{base_fn}.{name}.{PLOTS_EXT}"
            print(f"Saving pot = {plot_fn}")
            fig.savefig(plot_fn)
