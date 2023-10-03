# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import json
import os
from argparse import ArgumentParser

import pandas as pd
from tabulate import tabulate

from nemo.collections.common.metrics.per import PERData
from nemo.utils import logging

"""
Script to compute rates of punctuation correctness and errors based on the provided punctuation marks 
and the given manifest in .json file format.

# Arguments
    
    input_manifest_path: str, Required - path to dataset .json manifest file with stored reference text and ASR output text.
    punctuation_marks:, list[str], Required -  list of punctuation marks for estimating correctness and error rates among them.
    reference_field: str, Optional - name of field in .json manifest with the reference text ("text" by default).
    hypothesis_field: str, Optional - name of field in .json manifest with the hypothesis text ("pred_text" by default).
    punctuation_mask: str, Optional - mask token that will be applied to given punctuation marks while edit distance is calculated ("[PUNCT]" by default).
    output_manifest_path: str, Optional - path where .json manifest with calculated metrics will be saved.

# Usage
To score a dataset with a manifest file and given punctuation marks list:

python rate_punctuation.py \
    --input_manifest_path <Mandatory: Path to an ASR dataset manifest file> \
    --punctuation_marks . , ? \
    --reference_field "text" \
    --hypothesis_field "pred_text" \
    --output_manifest_path <Optional: Some output filename which will hold manifest with calculated metrics> 
"""

parser = ArgumentParser("Calculates Punctuaton Prediction Accuracy Rates")

parser.add_argument(
    "--manifest",
    required=True,
    type=str,
    help="Path .json manifest file with ASR predictions saved at `pred_text` field.",
)
parser.add_argument(
    "--punctuation_marks", required=True, nargs='+', help="Punctuation marks for calculating rates",
)
parser.add_argument(
    "--reference_field", default="text", type=str, help="Manifest field of reference text",
)
parser.add_argument(
    "--hypothesis_field", default="pred_text", type=str, help="Manifest field of hypothesis text",
)
parser.add_argument(
    "--output_manifest_path",
    default=None,
    type=str,
    help="Path where output .json manifest file \
    with metrics will be saved",
)
parser.add_argument(
    "--punctuation_mask",
    default="[PUNCT]",
    type=str,
    help="Mask template for substituting punctuation\
    marks during PER calculation",
)


def read_manifest(input_manifest_path: str) -> list[dict]:

    '''
    Reads .json format manifest and returns samples list where each sample is a dict.

    Args: 
        input_manifest_path (str) - path .json manifest file with ASR predictions saved at `pred_text` field.
    
    Returns:
        samples - list of dict
    '''

    assert os.path.exists(input_manifest_path), f"Input manifest file is not found: {input_manifest_path}"

    with open(input_manifest_path, "r") as manifest:
        lines = manifest.readlines()
        samples = [json.loads(line) for line in lines]
        return samples


def write_manifest(output_manifest_path: str, samples: list[dict]) -> None:

    assert os.path.exists(
        os.path.dirname(output_manifest_path)
    ), f"Directory to save output manifest path does not exists: {output_manifest_path}"

    '''
    Writes samples to .json file.
    
    Args:
        output_manifest_path (str) - path where .json manifest with calculated metrics
        will be saved
        
        samples (list[dict]) - list of samples 
    '''

    with open(output_manifest_path, "w") as output:
        for sample in samples:
            line = json.dumps(sample)
            output.writelines(f'{line}\n')
    logging.info(f'Output manifest saved: {output_manifest_path}')


def compute_rates(
    input_manifest_path: str,
    punctuation_marks: list[str],
    reference_field: str = "text",
    hypothesis_field: str = "pred_text",
    punctuation_mask: str = "[PUNCT]",
    output_manifest_path: str = None,
) -> None:

    '''
    Calculates punctuation correctness and error rates based on the provided punctuation marks 
    and the given manifest.json file.
    
    Args:
        input_manifest_path (str): path .json manifest file with ASR predictions saved at `pred_text` field.
        punctuation_marks (list[str]): list of punctuation marks for metrics computing
        reference_field (str): name of field in .json manifest with the reference text ("text" by default)
        hypothesis_field (str): name of field in .json manifest with the hypothesis text ("pred_text" by default)
        punctuation_mask (str): mask token that will be applied to given punctuation marks while edit distance is calculated ("[PUNCT]" by default)
        output_manifest_path (str): path where .json manifest with calculated metrics will be saved
    '''

    samples = read_manifest(input_manifest_path)

    references, hypotheses = [], []

    for sample in samples:
        references.append(sample[reference_field])
        hypotheses.append(sample[hypothesis_field])

    logging.info(f'Rates Computing..:')

    per_data = PERData(
        references=references,
        hypotheses=hypotheses,
        punctuation_marks=punctuation_marks,
        punctuation_mask=punctuation_mask,
    )

    per_data.compute()

    if output_manifest_path is not None:
        i = 0
        while i < len(references):
            samples[i]['correct_rate'] = per_data.rates[i].correct_rate
            samples[i]['deletions_rate'] = per_data.rates[i].deletions_rate
            samples[i]['insertions_rate'] = per_data.rates[i].insertions_rate
            samples[i]['substitutions_rate'] = per_data.rates[i].substitution_rate
            samples[i]['per'] = per_data.rates[i].per

            i += 1

        write_manifest(output_manifest_path=output_manifest_path, samples=samples)

    logging.info(f'Correct rate: {per_data.correct_rate:2f}')
    logging.info(f'Deletions rate: {per_data.deletions_rate:2f}')
    logging.info(f'Insertions rate: {per_data.insertions_rate:2f}')
    logging.info(f'Substitutions rate: {per_data.substitution_rate:2f}')
    logging.info(f'Punctuation Error Rate: {per_data.per:2f}')

    rates_by_pm_df = pd.DataFrame(per_data.operation_rates)
    substitution_rates_by_pm_df = pd.DataFrame(per_data.substitution_rates)

    logging.info(
        "Rates of punctuation correctness and errors:\n" + tabulate(rates_by_pm_df, headers='keys', tablefmt='psql')
    )
    logging.info(
        "Substitution rates between puncutation marks:\n"
        + tabulate(substitution_rates_by_pm_df, headers='keys', tablefmt='psql')
    )


if __name__ == "__main__":
    args = parser.parse_args()

    compute_rates(
        input_manifest_path=args.manifest,
        punctuation_marks=args.punctuation_marks,
        reference_field=args.reference_field,
        hypothesis_field=args.hypothesis_field,
        punctuation_mask=args.punctuation_mask,
        output_manifest_path=args.output_manifest_path,
    )
