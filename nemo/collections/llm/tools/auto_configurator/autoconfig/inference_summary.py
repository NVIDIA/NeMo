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

import argparse
import csv
import glob
import os


def scrap_latency(file_name):
    with open(file_name, "r") as file_handle:
        for line in file_handle:
            if line.find("FT-CPP-decoding-beamsearch-time") != -1:
                elements = line.split(" ")
                return float(elements[-2])
    return "FAILURE"


def main():
    parser = argparse.ArgumentParser(
        description="Generate summary from inference benchmark"
    )
    parser.add_argument("--model-prefix", help="File prefix for logs", required=True)
    parser.add_argument(
        "--configs-csv",
        help="Path to CSV file with profile configurations",
        required=True,
    )
    parser.add_argument("--workspace", help="Path to workspace folder", required=True)
    parser.add_argument("--output", help="Path to save output summary", required=True)
    args = parser.parse_args()

    with open(args.configs_csv, "r") as csv_file:
        config_lines = list(csv.reader(csv_file))

    rows = []

    for tp, pp, bs in [l for l in config_lines[1:] if len(l) == 3]:
        file_prefix = (
            f"{args.workspace}/{args.model_prefix}_tp{tp}_pp{pp}_bs{bs}/log_job*.out"
        )
        files = [f for f in glob.glob(file_prefix) if os.path.isfile(f)]
        if len(files) != 1:
            latency = "MISSING_LOG"
        else:
            latency = scrap_latency(files[0])
        gpu_norm_throughput = round(
            int(bs) * 1000.0 / float(latency) / int(tp) / int(pp), 3
        )
        row = [tp, pp, bs, latency, gpu_norm_throughput]
        rows.append(row)

    header = ["TP", "PP", "BS", "Latency [ms]", "Throughput per GPU [samples/sec/gpu]"]

    with open(args.output, "w") as output_file:
        output_writer = csv.writer(output_file)
        output_writer.writerow(header)
        output_writer.writerows(rows)


if __name__ == "__main__":
    main()
