# Copyright (c) 2019 NVIDIA Corporation
import os

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from nemo.utils.exp_logging import get_logger

logger = get_logger('')

def eval_iter_callback(tensors, global_vars, eval_data_layer, tag_ids):
    if "correct_tags" not in global_vars.keys():
        global_vars["correct_tags"] = 0
    if "total_tags" not in global_vars.keys():
        global_vars["total_tags"] = 0
    if "predicted_tags" not in global_vars.keys():
        global_vars["predicted_tags"] = 0
    if "correct_labels" not in global_vars.keys():
        global_vars["correct_labels"] = 0
    if "token_count" not in global_vars.keys():
        global_vars["token_count"] = 0
    if "lines" not in global_vars.keys():
        global_vars["lines"] = []

    lines = global_vars["lines"]

    logits_lists = []
    seq_ids = []

    for kv, v in tensors.items():
        if 'logits' in kv:
            for v_tensor in v:
                for logit_tensor in v_tensor:
                    logits_lists.append(logit_tensor.detach().cpu().tolist())

        if 'seq_ids' in kv:
            for v_tensor in v:
                for seq_id_tensor in v_tensor:
                    seq_ids.append(seq_id_tensor.detach().cpu().tolist())

    correct_tags, total_tags, predicted_tags, correct_labels, token_count, \
        lines = eval_data_layer.eval_preds(logits_lists, seq_ids, tag_ids)

    global_vars["correct_tags"] += correct_tags
    global_vars["total_tags"] += total_tags
    global_vars["predicted_tags"] += predicted_tags
    global_vars["correct_labels"] += correct_labels
    global_vars["token_count"] += token_count
    global_vars["lines"].extend(lines)


def eval_epochs_done_callback(global_vars, tag_ids, output_filename):
    correct_tags = global_vars["correct_tags"]
    total_tags = global_vars["total_tags"]
    predicted_tags = global_vars["predicted_tags"]
    correct_labels = global_vars["correct_labels"]
    token_count = global_vars["token_count"]
    lines = global_vars["lines"]

    if output_filename is not None:
        # Create output file
        tag_ids = {tag_ids[k]: k for k in tag_ids}

        last_label = ""
        last_prediction = ""

        with open(output_filename, "w") as f:
            for line in lines:
                if line["word"] == "":
                    f.write("\n")
                    last_label = ""
                    last_prediction = ""
                    continue

                label = tag_ids[int(line["label"])]
                prediction = tag_ids[int(line["prediction"])]

                last_label = line["label"]
                last_prediction = line["prediction"]

                f.write("{}\t{}\t{}\n".format(line["word"], label, prediction))

    accuracy = correct_labels / token_count

    p = correct_tags / predicted_tags if predicted_tags > 0 else 0
    r = correct_tags / total_tags if total_tags > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0

    results = {"Accuracy": accuracy, "f1": f1}
    logger.info(results)
    
    
    df = pd.read_csv(output_filename, header=None, sep='\t')
    df.columns = ['word', 'labels', 'preds']
    logger.info(classification_report(df.labels, df.preds, labels=list(tag_ids.values())))
    
    return results
