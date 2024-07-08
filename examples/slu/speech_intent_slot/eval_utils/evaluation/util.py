import json
import logging
import os
from typing import Any, Dict, Tuple

import tabulate
from progress.bar import Bar

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
)

logger = logging.getLogger(__name__)

SEPARATORS = {"csv": ",", "tsv": "\t"}


def load_predictions(path: str, load_gold: bool = False):
    """
    Load prediction file in a single dictionary, indexed by filename.

    :param path: Path to prediction file
    :param load_gold: When evaluating gold hypotheses upper bound
    :return: Dictionary of predictions
    """
    data_path = os.path.join(path)
    result = {}
    with open(data_path, "r") as f:
        lines = list(f)
        bar = Bar(message="Loading prediction file", max=len(lines))
        for line in lines:
            example = json.loads(line)
            result[example.pop("slurp_id" if load_gold else "file")] = example
            bar.next()
        bar.finish()
    return result


def load_gold_data(path: str, load_gold: bool = False):
    """
    Load gold file (test.jsonl) in a single dictionary, indexed by filename.

    :param path: Path to gold file
    :param load_gold: When evaluating gold hypotheses upper bound
    :return: Dictionary of gold examples
    """
    data_path = os.path.join(path)
    result = {}
    with open(data_path, "r") as f:
        lines = list(f)
        bar = Bar(message="Loading gold data", max=len(lines))
        for line in lines:
            example = json.loads(line)
            result.update(release2prediction(example, load_gold))
            bar.next()
        bar.finish()
    return result


def release2prediction(example: Dict[str, Any], load_gold: bool = False):
    """
    Convert the SLURP release format into prediction format.

    :param example: the example in release format (as they come with the dataset release)
    :param load_gold: When evaluating gold hypotheses upper bound
    :return: a list of examples in prediction format: List[Dict[str, Union[str, List]]]
    """
    result = {}
    res = {
        "text": " ".join([t["surface"] for t in example["tokens"]]),
        "scenario": example["scenario"],
        "action": example["action"],
        "entities": [
            {
                "type": entity["type"],
                "filler": " ".join([example["tokens"][i]["surface"].lower() for i in entity["span"]]),
            }
            for entity in example["entities"]
        ],
    }
    if load_gold:
        result[str(example["slurp_id"])] = res
    else:
        for file in example["recordings"]:
            result[file["file"]] = res
    return result


def format_results(
    results: Dict[str, Tuple[float]],
    label: str,
    full: bool = True,
    errors: bool = False,
    table_layout: str = "fancy_grid",
):
    """
    Util to format and print the results.

    Format results in tabular format.
    :param results: the dictionary output by the get_metric() method
    :param label: the title of the table to print
    :param full: is true, prints the results of all the labels. Otherwise prints just the average among them
    :param errors: if true, prints TPs, FPs and FNs
    :param table_layout: the table layout. Available: all those from `tabulate`, `csv` and `tsv`.
    :return: the formatted table as string
    """
    if errors:
        threshold = 100
    else:
        threshold = 4
    header = [label.capitalize(), "Precision", "Recall", "F-Measure", "TP", "FP", "FN"][:threshold]
    table = [["overall".upper(), *results.pop("overall")][:threshold]]
    if full:
        for label in results:
            table.append([label, *results[label]][:threshold])
    if table_layout in {"csv", "tsv"}:
        for i, row in enumerate(table):
            for j, item in enumerate(row):
                table[i][j] = str(item)
        return (
            SEPARATORS[table_layout].join(header)
            + "\n"
            + "\n".join([SEPARATORS[table_layout].join(row) for row in table])
        )
    if table_layout not in tabulate.tabulate_formats:
        logger.warning("{} non valid as table format. Using ``fancy_grid``".format(table_layout))
        table_layout = "fancy_grid"

    return tabulate.tabulate(
        table, headers=header, tablefmt=table_layout, floatfmt=("", ".4f", ".4f", ".4f", ".0f", ".1f", ".1f")
    )
