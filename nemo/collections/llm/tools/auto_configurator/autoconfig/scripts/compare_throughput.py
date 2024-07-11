import csv
import os
import re
import sys
from shutil import copyfile

import hydra
import pandas as pd
from omegaconf import OmegaConf
from tensorboard.backend.event_processing import event_accumulator


@hydra.main(config_path="../../conf", config_name="config")
def main(cfg):
    auto_configurator_path = cfg.auto_configurator_path
    settings_cfg = cfg.search_config.train_settings
    model_size = settings_cfg.model_size_in_b
    output_top_n = settings_cfg.output_top_n
    nodes = settings_cfg.num_nodes

    training_logs = os.path.join(settings_cfg.get("logs"), "training_logs")
    candidate_configs = os.path.join(settings_cfg.get("logs"), "candidate_configs")
    final_result_logs = os.path.join(settings_cfg.get("logs"), "final_result")

    result_columns = [
        "Model Name",
        "Model Size",
        "Seq Length",
        "TP",
        "PP",
        "CP",
        "EP",
        "MBS",
        "Act Ckpt Layers",
        "Act Ckpt Micro Bathes",
        "Act Ckpt Layers per Pipeline",
        "Num Layers",
        "Hidden Size",
        "FFN Hidden Size",
        "GBS",
        "Nodes",
        "GPUs per Node",
        "Time per Step",
        "Samples per Second",
        "Model TFLOPS / GPU",
        "Model TFLOPS Aggregate",
        "Config Name",
    ]
    error_columns = [
        "Model Name",
        "Model Size",
        "Seq Length",
        "TP",
        "PP",
        "CP",
        "EP",
        "MBS",
        "Act Ckpt Layers",
        "Act Ckpt Micro Bathes",
        "Act Ckpt Layers per Pipeline",
        "Num Layers",
        "Hidden Size",
        "FFN Hidden Size",
        "GBS",
        "Nodes",
        "GPUs per Node",
        "Error Message",
    ]
    result = []
    errors = []
    dirs = os.listdir(training_logs)
    for candidate_dir in dirs:
        config_path = os.path.join(candidate_configs, f"{candidate_dir}.yaml")
        candidate_cfg = OmegaConf.load(config_path)
        model_cfg = candidate_cfg.get("model")
        encoder_cfg = model_cfg.get("encoder")
        decoder_cfg = model_cfg.get("decoder")
        data_cfg = model_cfg.get("data")
        trainer_cfg = candidate_cfg.get("trainer")

        model_name = candidate_cfg.get("run").get("name").split("_")[0]
        gbs = model_cfg.get("global_batch_size")
        enc_seq_len = (
            model_cfg.get("encoder_seq_length")
            if model_name in ("gpt3", "bert", "llama", "baichuan2", "chatglm", "qwen2", "mixtral")
            else model_cfg.get("seq_length")
        )
        dec_seq_len = data_cfg.get("seq_length_dec")

        if model_name in (
            "gpt3",
            "bert",
            "llama",
            "baichuan2",
            "chatglm",
            "qwen2",
            "mixtral",
        ):
            hs = model_cfg.get("hidden_size")
            ffn_hs = None
            layers = model_cfg.get("num_layers")
            act_ckpt_layers = model_cfg.get("activations_checkpoint_num_layers")
            num_mbs_act = model_cfg.get("num_micro_batches_with_partial_activation_checkpoints")
            act_per_pipe = model_cfg.get("activations_checkpoint_layers_per_pipeline")
            cp = model_cfg.get("context_parallel_size")
            ep = model_cfg.get("expert_model_parallel_size")
        else:
            hs = encoder_cfg.get("hidden_size")
            ffn_hs = encoder_cfg.get("ffn_hidden_size")
            layers = encoder_cfg.get("num_layers") + decoder_cfg.get("num_layers")
            act_ckpt_layers = encoder_cfg.get("activations_checkpoint_num_layers") + decoder_cfg.get(
                "activations_checkpoint_num_layers"
            )
            num_mbs_act = None
            act_per_pipe = None
            cp = None
            ep = None
        tp = model_cfg.get("tensor_model_parallel_size")
        pp = model_cfg.get("pipeline_model_parallel_size")
        mbs = model_cfg.get("micro_batch_size")
        vocab = settings_cfg.get("vocab_size")
        gpus_per_node = trainer_cfg.get("devices")

        if f"{nodes}nodes" not in candidate_dir:
            continue

        for f in os.listdir(os.path.join(training_logs, candidate_dir)):
            if f.endswith(".err"):
                error_file = os.path.join(training_logs, candidate_dir, f)
                error = find_error(error_file)
                if error:
                    errors.append(
                        [
                            model_name,
                            model_size,
                            enc_seq_len,
                            tp,
                            cp,
                            ep,
                            pp,
                            mbs,
                            act_ckpt_layers,
                            num_mbs_act,
                            act_per_pipe,
                            layers,
                            hs,
                            ffn_hs,
                            gbs,
                            nodes,
                            gpus_per_node,
                            error,
                        ]
                    )

        files = os.listdir(os.path.join(training_logs, candidate_dir, "results"))
        for f in files:
            if f[:6] == "events":
                event_file = os.path.join(training_logs, candidate_dir, "results", f)
                ea = event_accumulator.EventAccumulator(event_file)
                ea.Reload()
                try:
                    timing_list = ea.Scalars("train_step_timing in s")
                    if len(timing_list) <= 6:
                        continue
                    timing_list = [x.value for x in timing_list[5:]]
                    avg_global_step_time = round(sum(timing_list) / len(timing_list), 4)
                    samples_per_s = round(gbs / avg_global_step_time, 2)
                    m_tflops, m_tflops_gpu = calculate_tflops(
                        model_name=model_name,
                        gbs=gbs,
                        enc_seq_len=enc_seq_len,
                        dec_seq_len=dec_seq_len,
                        hs=hs,
                        ffn_hs=ffn_hs,
                        layers=layers,
                        vocab=vocab,
                        nodes=nodes,
                        gpus_per_node=gpus_per_node,
                        time_per_step=avg_global_step_time,
                    )
                    config_name = f"tp{tp}_pp{pp}_cp{cp}_ep{ep}_mbs{mbs}_act_{act_ckpt_layers}_num_mbs_act_{num_mbs_act}_act_per_pipe_{act_per_pipe}"
                    result.append(
                        [
                            model_name,
                            model_size,
                            enc_seq_len,
                            tp,
                            pp,
                            cp,
                            ep,
                            mbs,
                            act_ckpt_layers,
                            num_mbs_act,
                            act_per_pipe,
                            layers,
                            hs,
                            ffn_hs,
                            gbs,
                            nodes,
                            gpus_per_node,
                            avg_global_step_time,
                            samples_per_s,
                            m_tflops_gpu,
                            m_tflops,
                            config_name,
                        ]
                    )
                finally:
                    continue

    result.sort(key=lambda x: x[15])
    print(f"Top {min(output_top_n, len(result))} configs sorted from fastest to slowest:")
    for i, res in enumerate(result):
        print(f"Config #{i+1}: {res[-1]} with {res[14]:.4f}s per global step.")
        if i + 1 == output_top_n:
            break

    top_config = f"{model_name}_{model_size}b_{nodes}nodes_tp_{result[0][3]}_pp_{result[0][4]}_cp_{result[0][5]}_ep_{result[0][6]}_mbs_{result[0][7]}_act_ckpt_{result[0][8]}_num_mbs_act_{result[0][9]}_act_per_pipe_{result[0][10]}"
    print("\n==================================================")
    print(f"Optimal config: {top_config} with {result[0][14]:.4f}s per global step.")
    print(f"Saving config to {final_result_logs}/optimal_config_{model_size}b_{nodes}nodes.yaml.")
    print("==================================================\n")

    # Save results as a CSV file.
    os.makedirs(final_result_logs, exist_ok=True)
    result_df = pd.DataFrame(result, columns=result_columns)
    result_df.to_csv(os.path.join(final_result_logs, f"final_summary_{nodes}nodes.csv"), index=False)

    error_df = pd.DataFrame(errors, columns=error_columns)
    error_df.to_csv(os.path.join(final_result_logs, f"failed_jobs_{nodes}nodes.csv"), index=False)

    copyfile(
        os.path.join(candidate_configs, f"{top_config}.yaml"),
        os.path.join(final_result_logs, f"optimal_config_{model_size}b_{nodes}nodes.yaml"),
    )


def calculate_tflops(
    model_name,
    gbs,
    enc_seq_len,
    dec_seq_len,
    hs,
    ffn_hs,
    layers,
    vocab,
    nodes,
    gpus_per_node,
    time_per_step,
):
    """Calculates model and hardware TFLOPS for each model.

    GPT-3 Formulas:
        Model FLOPs = (24𝐵𝑠ℎ^2 + 4𝐵𝑠^2ℎ) x (3 x num_layers) + 6𝐵𝑠ℎ
    T5/mT5 Formula:
        Model FLOPs =
    Bert Formula:
        Model FLOPs = 72BLsh^2 * ( 1 + (s/6h) + (v/12hL))
    """
    if model_name in ["gpt3", "llama", "baichuan2", "chatglm", "qwen2", "mixtral"]:
        # Model FLOPS calculation
        model_flops = (
            (24 * gbs * enc_seq_len * hs * hs + 4 * gbs * enc_seq_len * enc_seq_len * hs) * (3 * layers)
            + (6 * gbs * enc_seq_len * hs * vocab)
        ) / time_per_step
        model_flops_per_gpu = model_flops / (nodes * gpus_per_node)

        model_tflops = model_flops / 1e12
        model_tflops_per_gpu = model_flops_per_gpu / 1e12

    elif model_name == "bert":
        model_flops = (
            72 * gbs * layers * enc_seq_len * hs * hs * (1 + (enc_seq_len / (6 * hs)) + (vocab / (12 * hs * layers)))
        ) / time_per_step
        model_flops_per_gpu = model_flops / (nodes * gpus_per_node)
        model_tflops = model_flops / 1e12
        model_tflops_per_gpu = model_flops_per_gpu / 1e12

    elif model_name in ["t5", "mt5"]:
        # Encoder Layer FLOPS: include self attention + MLP
        flops_self_attn_enc = 8 * gbs * enc_seq_len * hs * hs + 4 * gbs * enc_seq_len * enc_seq_len * hs
        flops_mlp_enc = 6 * gbs * enc_seq_len * hs * ffn_hs  # geglu needs two gemms for h -> ffn_h
        flops_enc_layer = flops_self_attn_enc + flops_mlp_enc

        # Decoder Layer FLOPS: inlcude self_attn + cross_attn + MLP
        flops_self_attn_dec = 8 * gbs * dec_seq_len * hs * hs + 4 * gbs * dec_seq_len * dec_seq_len * hs
        flops_cross_attn_dec = (
            4 * gbs * enc_seq_len * hs * hs
            + 4 * gbs * dec_seq_len * hs * hs
            + 4 * gbs * enc_seq_len * dec_seq_len * hs
        )
        flops_mlp_dec = 6 * gbs * dec_seq_len * hs * ffn_hs  # geglu needs two gemms for h -> ffn_h
        flops_dec_layer = flops_self_attn_dec + flops_cross_attn_dec + flops_mlp_dec

        # FLOPs of logits layer in the head
        flops_logits = 2 * gbs * dec_seq_len * hs * vocab

        # FLOPs of fprop
        flops_fprop = (flops_enc_layer + flops_dec_layer) * (layers // 2) + flops_logits

        # FLOPs of each train step (FLOPs of bprop is 2*fprop)
        model_flops = 3 * flops_fprop / time_per_step
        model_flops_per_gpu = model_flops / (nodes * gpus_per_node)
        model_tflops = model_flops / 1e12
        model_tflops_per_gpu = model_flops_per_gpu / 1e12

    else:
        raise NotImplementedError("Model type not supported.")
    return round(model_tflops, 2), round(model_tflops_per_gpu, 2)


def find_error(error_file: str, errors: list = ["CUDA out of memory"]):
    """
    Finds the error among job output.
    :param list errors: list of "popular" errors.
    :param str error_file: path to the job output.
    :return: str error if job has been failed because of one of listed errors and None if not.
    :rtype: str
    """
    error = None
    with open(error_file, "r") as f:
        output = f.read()
    for e in errors:
        if e in output:
            error = e
    return error


if __name__ == "__main__":
    main()
