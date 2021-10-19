#!/usr/bin/env python3
"""
This script takes as an input XXXX.json files
(i.e., the output of nmt_transformer_infer.py --write_timing)
and creates plots XXX.PLOT_NAME.png at the same path.
"""
from matplotlib import pyplot as plt
import json
import sys
import os

#=============================================================================#
# Helper functions
#=============================================================================#


def plot_timing(lengths, timings, lengths_name, timings_name, fig=None):
    if fig is None:
        fig = plt.figure()

    plt.scatter(lengths, timings, label=timings_name)
    plt.xlabel(f"{lengths_name} [tokens]")
    plt.ylabel(f"{timings_name} [sec]")
    plt.grid(True)
    plt.title(f"{timings_name} vs. {lengths_name}")

    return fig


#=============================================================================#
# Main script
#=============================================================================#
if __name__ == "__main__":
    for timing_fn in sys.argv[1:]:
        # load data
        print(f"Parsing file = {timing_fn}")
        all_data = json.load(open(timing_fn))

        # parse data
        data = {}
        for k in all_data[0].keys():
            data[k] = [d[k] for d in all_data]

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
            plot_fn = f"{base_fn}.{name}.png"
            print(f"Saving pot = {plot_fn}")
            fig.savefig(plot_fn)