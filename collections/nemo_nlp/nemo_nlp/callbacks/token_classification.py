# Copyright (c) 2019 NVIDIA Corporation


def eval_iter_callback(tensors, global_vars, eval_data_layer):
    if "correct_labels" not in global_vars.keys():
        global_vars["correct_labels"] = 0
    if "incorrect_labels" not in global_vars.keys():
        global_vars["incorrect_labels"] = 0
    if "correct_preds" not in global_vars.keys():
        global_vars["correct_preds"] = 0
    if "total_preds" not in global_vars.keys():
        global_vars["total_preds"] = 0
    if "total_correct" not in global_vars.keys():
        global_vars["total_correct"] = 0

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

    correct_labels, incorrect_labels, correct_preds, total_preds, \
        total_correct = eval_data_layer.eval_preds(logits_lists, seq_ids)

    global_vars["correct_labels"] += correct_labels
    global_vars["incorrect_labels"] += incorrect_labels
    global_vars["correct_preds"] += correct_preds
    global_vars["total_preds"] += total_preds
    global_vars["total_correct"] += total_correct


def eval_epochs_done_callback(global_vars):

    correct_labels = global_vars["correct_labels"]
    incorrect_labels = global_vars["incorrect_labels"]
    correct_preds = global_vars["correct_preds"]
    total_preds = global_vars["total_preds"]
    total_correct = global_vars["total_correct"]

    accuracy = correct_labels / (correct_labels + incorrect_labels)

    p = correct_preds / total_preds
    r = correct_preds / total_correct
    f1 = 2 * p * r / (p + r)

    print(f"Accuracy = {accuracy}")
    print(f"F1= {f1}")

    return dict({"accuracy": accuracy, "f1": f1})
