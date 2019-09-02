# Copyright (c) 2019 NVIDIA Corporation


def eval_iter_callback(tensors, global_vars, eval_data_layer, tag_ids):
    if "correct_tags" not in global_vars.keys():
        global_vars["correct_tags"] = 0
    if "token_count" not in global_vars.keys():
        global_vars["token_count"] = 0
    if "correct_chunks" not in global_vars.keys():
        global_vars["correct_chunks"] = 0
    if "predicted_chunks" not in global_vars.keys():
        global_vars["predicted_chunks"] = 0
    if "total_chunks" not in global_vars.keys():
        global_vars["total_chunks"] = 0
    if "lines" not in global_vars.keys():
        global_vars["lines"] = []

    logits_lists = []
    seq_ids = []

    for kv, v in tensors.items():
        if "logits" in kv:
            for v_tensor in v:
                for logit_tensor in v_tensor:
                    logits_lists.append(logit_tensor.detach().cpu().tolist())

        if "seq_ids" in kv:
            for v_tensor in v:
                for seq_id_tensor in v_tensor:
                    seq_ids.append(seq_id_tensor.detach().cpu().tolist())

    correct_tags, token_count, correct_chunks, predicted_chunks, \
        total_chunks, lines = \
        eval_data_layer.eval_preds(logits_lists, seq_ids, tag_ids)

    global_vars["correct_tags"] += correct_tags
    global_vars["token_count"] += token_count
    global_vars["correct_chunks"] += correct_chunks
    global_vars["predicted_chunks"] += predicted_chunks
    global_vars["total_chunks"] += total_chunks
    global_vars["lines"].extend(lines)


def eval_epochs_done_callback(global_vars, tag_ids, output_filename):
    correct_tags = global_vars["correct_tags"]
    token_count = global_vars["token_count"]
    correct_chunks = global_vars["correct_chunks"]
    predicted_chunks = global_vars["predicted_chunks"]
    total_chunks = global_vars["total_chunks"]
    lines = global_vars["lines"]

    if output_filename is not None:
        # Create output file that can be evaluated by conlleval.pl script
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

                # Correctly precede tags with B- and I- as necessary (slightly
                # modified from https://www.clips.uantwerpen.be/conll2003/ner/)
                if label != "O":
                    if last_label == line["label"]:
                        label = "I-{}".format(label)
                    else:
                        label = "B-{}".format(label)

                if prediction != "O":
                    if last_prediction == line["prediction"]:
                        prediction = "I-{}".format(prediction)
                    else:
                        prediction = "B-{}".format(prediction)

                last_label = line["label"]
                last_prediction = line["prediction"]

                f.write("{}\t{}\t{}\n".format(line["word"], label, prediction))

    accuracy = correct_tags / token_count

    p = correct_chunks / predicted_chunks if predicted_chunks > 0 else 0
    r = correct_chunks / total_chunks if total_chunks > 0 else 0
    f1 = 2 * p * r / (p + r) if p > 0 and r > 0 else 0

    print(f"Accuracy = {accuracy}")
    print(f"F1 = {f1}")

    return {"accuracy": accuracy, "f1": f1}
