import json
import re
from argparse import ArgumentParser

import pandas as pd
from tabulate import tabulate
from tqdm import tqdm


class PER:
    punctuation_marks = None
    punctuation_mask = "[PUNCT]"

    def __init__(self, punctuation_marks: list[str], punctuation_mask: str = "[PUNCT]"):
        PER.punctuation_marks = punctuation_marks
        PER.punctuation_mask = punctuation_mask

    class Statistics:
        def __init__(self) -> None:
            self.punctuation_marks_statistics = pd.DataFrame(
                0, index=["Correct", "Deletions", "Insertions", "Substitutions"], columns=PER.punctuation_marks
            )

            self.substitution_statistics = pd.DataFrame(0, index=PER.punctuation_marks, columns=PER.punctuation_marks)

            self.punctuation_marks_rates = pd.DataFrame(
                0, index=["Correct", "Deletions", "Insertions", "Substitutions", "PER"], columns=PER.punctuation_marks
            )

            self.substitution_rates = pd.DataFrame(0, index=PER.punctuation_marks, columns=PER.punctuation_marks)

            self.correct_rate = 0
            self.deletions_rate = 0
            self.insertions_rate = 0
            self.substitutions_rate = 0
            self.per = 0

        def update_statistics(self, per_statistics_obj: object):
            self.punctuation_marks_statistics = self.punctuation_marks_statistics.add(
                per_statistics_obj.punctuation_marks_statistics, fill_value=0
            )

            self.substitution_statistics = self.substitution_statistics.add(
                per_statistics_obj.substitution_statistics, fill_value=0
            )

        def compute_rates(self) -> None:
            amounts_sum = 0

            for punctuation_mark in PER.punctuation_marks:
                punctuation_mark_amounts_sum = self.punctuation_marks_statistics[punctuation_mark].sum()
                if punctuation_mark_amounts_sum == 0:
                    continue

                amounts_sum += punctuation_mark_amounts_sum
                self.punctuation_marks_rates[punctuation_mark] = self.punctuation_marks_statistics[
                    punctuation_mark
                ].div(punctuation_mark_amounts_sum)
                self.substitution_rates[punctuation_mark] = self.substitution_statistics[punctuation_mark].div(
                    punctuation_mark_amounts_sum
                )

                self.punctuation_marks_rates.loc["PER", punctuation_mark] = (
                    self.punctuation_marks_rates.loc["Deletions", punctuation_mark]
                    + self.punctuation_marks_rates.loc["Insertions", punctuation_mark]
                    + self.punctuation_marks_rates.loc["Substitutions", punctuation_mark]
                )

            if amounts_sum == 0:
                return

            self.correct_rate = self.punctuation_marks_statistics.loc["Correct"].sum() / amounts_sum
            self.deletions_rate = self.punctuation_marks_statistics.loc["Deletions"].sum() / amounts_sum
            self.insertions_rate = self.punctuation_marks_statistics.loc["Insertions"].sum() / amounts_sum
            self.substitutions_rate = self.punctuation_marks_statistics.loc["Substitutions"].sum() / amounts_sum
            self.per = self.deletions_rate + self.insertions_rate + self.substitutions_rate

            return

    class Sample:
        def __init__(self, reference: str, hypothesis: str) -> None:
            self.reference = reference
            self.hypothesis = hypothesis

            self.statistics = PER.Statistics()

        def _tokenize(self, text: str):
            punctuation_marks = "\\" + "\\".join(PER.punctuation_marks)
            tokens = re.findall(rf"[\w']+|[{punctuation_marks}]", text)
            return tokens

        def _mask_punct_tokens(self, tokens: list[str]):
            masked = [PER.punctuation_mask if token in PER.punctuation_marks else token for token in tokens]
            return masked

        def compute_operations_amounts(self):
            r_tokens = self._tokenize(self.reference)
            h_tokens = self._tokenize(self.hypothesis)

            r_masked = self._mask_punct_tokens(r_tokens)
            h_masked = self._mask_punct_tokens(h_tokens)

            r_punct_amount = r_masked.count(PER.punctuation_mask)
            h_punct_amount = h_masked.count(PER.punctuation_mask)

            if r_punct_amount + h_punct_amount == 0:
                return
            else:
                r_len = len(r_masked)
                h_len = len(h_masked)

                costs = [[0 for inner in range(h_len + 1)] for outer in range(r_len + 1)]
                backtrace = [[0 for inner in range(h_len + 1)] for outer in range(r_len + 1)]

                OP_COR = 'C'
                OP_SUB = 'S'
                OP_INS = 'I'
                OP_DEL = 'D'

                DEL_PENALTY = 1
                INS_PENALTY = 1
                SUB_PENALTY = 1

                for i in range(1, r_len + 1):
                    costs[i][0] = DEL_PENALTY * i
                    backtrace[i][0] = OP_DEL

                for j in range(1, h_len + 1):
                    costs[0][j] = INS_PENALTY * j
                    backtrace[0][j] = OP_INS

                for j in range(1, h_len + 1):
                    costs[0][j] = INS_PENALTY * j
                    backtrace[0][j] = OP_INS

                for i in range(1, r_len + 1):
                    for j in range(1, h_len + 1):
                        if r_masked[i - 1] == h_masked[j - 1]:
                            costs[i][j] = costs[i - 1][j - 1]
                            backtrace[i][j] = OP_COR
                        else:
                            substitutionCost = costs[i - 1][j - 1] + SUB_PENALTY
                            insertionCost = costs[i][j - 1] + INS_PENALTY
                            deletionCost = costs[i - 1][j] + DEL_PENALTY

                            costs[i][j] = min(substitutionCost, insertionCost, deletionCost)
                            if costs[i][j] == substitutionCost:
                                backtrace[i][j] = OP_SUB
                            elif costs[i][j] == insertionCost:
                                backtrace[i][j] = OP_INS
                            else:
                                backtrace[i][j] = OP_DEL

                i = r_len
                j = h_len

                while i > 0 or j > 0:
                    if backtrace[i][j] == OP_COR:
                        if r_masked[i - 1] == PER.punctuation_mask or h_masked[j - 1] == PER.punctuation_mask:
                            r_token = r_tokens[i - 1]
                            h_token = h_tokens[j - 1]

                            if r_token == h_token:
                                self.statistics.punctuation_marks_statistics.loc['Correct', r_token] += 1
                            else:
                                self.statistics.substitution_statistics.loc[r_token, h_token] += 1
                                self.statistics.punctuation_marks_statistics.loc['Substitutions', r_token] += 1
                        i -= 1
                        j -= 1

                    elif backtrace[i][j] == OP_SUB:
                        i -= 1
                        j -= 1

                    elif backtrace[i][j] == OP_INS:
                        j -= 1

                    elif backtrace[i][j] == OP_DEL:
                        i -= 1

                for punctuation_mark in PER.punctuation_marks:
                    num_correct_of_punctution_mark = self.statistics.punctuation_marks_statistics.loc[
                        'Correct', punctuation_mark
                    ]

                    num_substitutions_of_punctuation_mark = self.statistics.substitution_statistics.loc[
                        punctuation_mark
                    ].sum()
                    num_substitutions_to_punctuation_mark = self.statistics.substitution_statistics.loc[
                        :, punctuation_mark
                    ].sum()

                    num_deletions_of_mark_per_sample = r_tokens.count(punctuation_mark) - (
                        num_correct_of_punctution_mark + num_substitutions_of_punctuation_mark
                    )
                    num_insertions_of_mark_per_sample = h_tokens.count(punctuation_mark) - (
                        num_correct_of_punctution_mark + num_substitutions_to_punctuation_mark
                    )

                    self.statistics.punctuation_marks_statistics.loc[
                        'Deletions', punctuation_mark
                    ] = num_deletions_of_mark_per_sample
                    self.statistics.punctuation_marks_statistics.loc[
                        'Insertions', punctuation_mark
                    ] = num_insertions_of_mark_per_sample
                    self.statistics.punctuation_marks_statistics.loc[
                        'Substitutions', punctuation_mark
                    ] = num_substitutions_of_punctuation_mark

                return

    class Dataset:
        def __init__(
            self,
            input_manifest_path: str,
            reference_field: str = "text",
            hypothesis_field: str = "pred_text",
            output_manifest_path: str = None,
        ) -> None:

            self.input_manifest_path = input_manifest_path
            self.reference_field = reference_field
            self.hypothesis_field = hypothesis_field
            self.output_manifest_path = output_manifest_path
            self.samples = []

            self.statistics = PER.Statistics()

        def read_manifest(self):
            with open(self.input_manifest_path, 'r') as manifest:
                lines = manifest.readlines()
                for line in tqdm(lines, desc="Reading manifest.."):
                    sample = json.loads(line)

                    sample = PER.Sample(
                        reference=sample[self.reference_field], hypothesis=sample[self.hypothesis_field]
                    )

                    self.samples.append(sample)

        def compute(self):
            for sample in tqdm(self.samples, desc="Computing.."):
                sample.compute_operations_amounts()
                sample.statistics.compute_rates()
                self.statistics.update_statistics(sample.statistics)

            self.statistics.compute_rates()

        def write_manifest(self):
            with open(self.output_manifest_path, 'w') as manifest:
                for sample in tqdm(self.samples, desc="Writing manifest"):
                    sample_dict = {
                        self.reference_field: sample.reference,
                        self.hypothesis_field: sample.hypothesis,
                        "correct_rate": sample.statistics.correct_rate,
                        "deletions_rate": sample.statistics.deletions_rate,
                        "insertions_rate": sample.statistics.insertions_rate,
                        "substitutions_rate": sample.statistics.substitutions_rate,
                        "per": sample.statistics.per,
                    }

                    line = json.dumps(sample_dict)
                    manifest.writelines(f'{line}\n')

        def print_rates(self):

            print("-" * 40)

            correct_rate = self.statistics.correct_rate
            print(f"Correct rate: {correct_rate} ({correct_rate * 100:2f}%)")

            deletions_rate = self.statistics.deletions_rate
            print(f"Deletions rate: {deletions_rate} ({deletions_rate * 100:2f}%)")

            insertions_rate = self.statistics.insertions_rate
            print(f"Insertions rate: {insertions_rate} ({insertions_rate * 100:2f}%)")

            substitutions_rate = self.statistics.substitutions_rate
            print(f"Substitutions rate: {substitutions_rate} ({substitutions_rate * 100:2f}%)")

            per = self.statistics.per
            print(f"Punctuation Error Rate: {per} ({per * 100:2f}%)")

            print("\nRATES BY PUNCTUATION MARK:")
            print(tabulate(self.statistics.punctuation_marks_rates, headers='keys', tablefmt='psql'))


if __name__ == "__main__":
    parser = ArgumentParser("Calculate Punctuaton Prediction Accuracy Rates")

    parser.add_argument(
        "--manifest",
        required=True,
        type=str,
        help="Path .json manifest file with ASR predictions saved at `pred_text` field.",
    )
    parser.add_argument(
        "--punctuation_marks",
        required=True,
        type=str,
        help="String of punctuation marks for calculating rates \
        separeted by vertical bar (ex. \".|,|?\")",
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

    args = parser.parse_args()
    args.punctuation_marks = args.punctuation_marks.split("|")

    per = PER(punctuation_marks=args.punctuation_marks, punctuation_mask=args.punctuation_mask)

    per_dataset = per.Dataset(
        input_manifest_path=args.manifest,
        reference_field=args.reference_field,
        hypothesis_field=args.hypothesis_field,
        output_manifest_path=args.output_manifest_path,
    )

    per_dataset.read_manifest()
    per_dataset.compute()

    if args.output_manifest_path is not None:
        per_dataset.write_manifest()

    per_dataset.print_rates()
