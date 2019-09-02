# Copyright (c) 2019 NVIDIA Corporation


def eval_iter_callback(tensors, global_vars):
    if "eval_start_logits" not in global_vars.keys():
        global_vars["eval_start_logits"] = []
    if "eval_end_logits" not in global_vars.keys():
        global_vars["eval_end_logits"] = []
    if "eval_unique_ids" not in global_vars.keys():
        global_vars["eval_unique_ids"] = []

    for kv, v in tensors.items():

        if 'logits' in kv:
            logits = []
            for v_tensor in v:
                for logit_tensor in v_tensor:
                    logits.append(logit_tensor.detach().cpu().tolist())

            if kv.startswith('start_logits'):
                global_vars['eval_start_logits'].extend(logits)
            elif kv.startswith('end_logits'):
                global_vars['eval_end_logits'].extend(logits)

        if 'unique_ids' in kv:
            unique_ids = []
            for v_tensor in v:
                for id_tensor in v_tensor:
                    unique_ids.append(id_tensor.detach().cpu().tolist())
            global_vars['eval_unique_ids'].extend(unique_ids)


def eval_epochs_done_callback(global_vars, eval_data_layer, do_lower_case):

    exact_match, f1 = eval_data_layer.calculate_exact_match_and_f1(
        global_vars["eval_unique_ids"],
        global_vars["eval_start_logits"],
        global_vars["eval_end_logits"],
        do_lower_case=do_lower_case)

    print(f"Exact_match = {exact_match}, f1 = {f1}")

    global_vars["eval_unique_ids"] = []
    global_vars["eval_start_logits"] = []
    global_vars["eval_end_logits"] = []

    return dict({"exact_match": exact_match, "f1": f1})
