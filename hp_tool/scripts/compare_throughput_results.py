import os
import csv
import sys
import re
from shutil import copyfile

import hydra
import pandas as pd
from omegaconf import OmegaConf
from tensorboard.backend.event_processing import event_accumulator


@hydra.main(config_path="../../conf", config_name="config")
def main(cfg):
    bignlp_hp_tool_path = cfg.bignlp_hp_tool_path
    hp_cfg = cfg.search_config
    settings_cfg = hp_cfg.train_settings
    model_size = settings_cfg.model_size_in_b
    output_top_n = settings_cfg.output_top_n
    nodes = cfg.get("nodes")

    training_logs = os.path.join(settings_cfg.get("logs"), "training_logs")
    candidate_configs = os.path.join(settings_cfg.get("logs"), "candidate_configs")
    final_result_logs = os.path.join(settings_cfg.get("logs"), "final_result")

    columns = [
        "Model Name",
        "Model Size",
        "TP",
        "PP",
        "MBS",
        "Act Ckpt Layers",
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
        "HW TFLOPS / GPU",
        "HW TFLOPS Aggregate",
        "Config Name",
    ]
    result = []
    dirs = os.listdir(training_logs)
    for candidate_dir in dirs:
        config_path = os.path.join(candidate_configs, f"{candidate_dir}.yaml")
        candidate_cfg = OmegaConf.load(config_path)
        model_cfg = candidate_cfg.get("model")
        data_cfg = model_cfg.get("data")
        trainer_cfg = candidate_cfg.get("trainer")

        model_name = candidate_cfg.get("run").get("name").split("_")[0]
        gbs = model_cfg.get("global_batch_size")
        enc_seq_len = (
            model_cfg.get("encoder_seq_length")
            if model_name == "gpt3"
            else model_cfg.get("seq_length")
        )
        dec_seq_len = data_cfg.get("seq_length_dec")
        hs = model_cfg.get("hidden_size")
        if model_name == "gpt3":
            ffn_hs = None
        else:
            ffn_hs = model_cfg.get("ffn_hidden_size")
        layers = model_cfg.get("num_layers")
        tp = model_cfg.get("tensor_model_parallel_size")
        pp = model_cfg.get("pipeline_model_parallel_size")
        mbs = model_cfg.get("micro_batch_size")
        act_ckpt_layers = model_cfg.get("activations_checkpoint_num_layers")
        vocab = settings_cfg.get("vocab_size")
        gpus_per_node = trainer_cfg.get("devices")

        if f"{nodes}nodes" not in candidate_dir:
            continue

        files = os.listdir(os.path.join(training_logs, candidate_dir))
        for f in files:
            if f[:6] == "events":
                event_file = os.path.join(training_logs, candidate_dir, f)
                ea = event_accumulator.EventAccumulator(event_file)
                ea.Reload()
                try:
                    timing_list = ea.Scalars("train_step_timing")
                    if len(timing_list) <= 6:
                        continue
                    timing_list = [x.value for x in timing_list[5:]]
                    avg_global_step_time = round(sum(timing_list) / len(timing_list), 4)
                    samples_per_s = round(gbs / avg_global_step_time, 2)
                    m_tflops, m_tflops_gpu, hw_tflops, hw_tflops_gpu = calculate_tflops(
                        model_name=model_name,
                        gbs=gbs,
                        enc_seq_len=enc_seq_len,
                        dec_seq_len=dec_seq_len,
                        hs=hs,
                        ffn_hs=ffn_hs,
                        layers=layers,
                        act_ckpt_layers=act_ckpt_layers,
                        vocab=vocab,
                        nodes=nodes,
                        gpus_per_node=gpus_per_node,
                        time_per_step=avg_global_step_time,
                    )
                    config_name = f"tp{tp}_pp{pp}_mbs{mbs}_act{act_ckpt_layers}"
                    result.append(
                        [
                            model_name,
                            model_size,
                            tp,
                            pp,
                            mbs,
                            act_ckpt_layers,
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
                            hw_tflops_gpu,
                            hw_tflops,
                            config_name,
                        ]
                    )
                finally:
                    continue

    result.sort(key=lambda x: x[12])
    print(
        f"Top {min(output_top_n, len(result))} configs sorted from fastest to slowest:"
    )
    for i, res in enumerate(result):
        print(f"Config #{i+1}: {res[-1]} with {res[12]:.4f}s per global step.")
        if i + 1 == output_top_n:
            break

    top_config = f"{model_name}_{model_size}b_{nodes}nodes_tp_{result[0][2]}_pp_{result[0][3]}_mbs_{result[0][4]}_act_ckpt_{result[0][5]}"
    print("\n==================================================")
    print(f"Optimal config: {top_config} with {result[0][12]:.4f}s per global step.")
    print(f"Saving config to {final_result_logs}/optimal_config_{model_size}b_{nodes}nodes.yaml.")
    print("==================================================\n")

    # Save results as a CSV file.
    os.makedirs(final_result_logs, exist_ok=True)
    df = pd.DataFrame(result, columns=columns)
    df.to_csv(os.path.join(final_result_logs, f"final_summary_{nodes}nodes.csv"), index=False)

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
    act_ckpt_layers,
    vocab,
    nodes,
    gpus_per_node,
    time_per_step,
):
    """Calculates model and hardware TFLOPS for each model.

    GPT-3 Formulas:
        Model FLOPs = (24𝐵𝑠ℎ^2 + 4𝐵𝑠^2ℎ) x (3 x num_layers) + 6𝐵𝑠ℎ
        HW FLOPs = (24𝐵𝑠ℎ^2 + 4𝐵𝑠^2ℎ) x (3 x num_layers + num_checkpt_layers) + 6𝐵𝑠ℎ
    T5/mT5 Formula:
        Model FLOPs = 
        HW FLOPs = 
        ((2*R3*M3*M3*(5*O3+4*P3)+6*R3*M3*N3*(O3+P3)+4*R3*M3*(O3*O3+P3*P3+O3*P3))*3*L3/2+6*R3*P3*M3*Q3)/(G3*H3)/1000000000000/F3
    """
    if model_name == "gpt3":
        # Model FLOPS calculation
        model_flops = ((
            24 * gbs * enc_seq_len * hs * hs + 4 * gbs * enc_seq_len * enc_seq_len * hs
        ) * (3 * layers) + (6 * gbs * enc_seq_len * hs * vocab)) / time_per_step
        model_flops_per_gpu = model_flops / (nodes * gpus_per_node)
        model_tflops = model_flops / 1e12
        model_tflops_per_gpu = model_flops_per_gpu / 1e12
        # HW FLOPS calculation
        hw_flops = ((24 * gbs * enc_seq_len * hs * hs + 4 * gbs * enc_seq_len * enc_seq_len * hs) * (
            3 * layers + act_ckpt_layers
        ) + (6 * gbs * enc_seq_len * hs * vocab)) / time_per_step
        hw_flops_per_gpu = hw_flops / (nodes * gpus_per_node)
        hw_tflops = hw_flops / 1e12
        hw_tflops_per_gpu = hw_flops_per_gpu / 1e12
    elif model_name in ["t5", "mt5"]:
        # Model FLOPS calculation
        model_flops = ((2*gbs*hs*hs * (5*enc_seq_len + 4*dec_seq_len) + 6*gbs*hs*ffn_hs*(enc_seq_len+dec_seq_len) + 4*gbs*hs*(enc_seq_len*enc_seq_len + dec_seq_len*dec_seq_len + enc_seq_len*dec_seq_len)) * 3*layers + 6*gbs*dec_seq_len*hs*vocab) / time_per_step
        model_flops_per_gpu = model_flops / (nodes * gpus_per_node)
        model_tflops = model_flops / 1e12
        model_tflops_per_gpu = model_flops_per_gpu / 1e12
        # HW FLOPS calculation
        hw_flops = ((2*gbs*hs*hs * (5*enc_seq_len + 4*dec_seq_len) + 6*gbs*hs*ffn_hs*(enc_seq_len+dec_seq_len) + 4*gbs*hs*(enc_seq_len*enc_seq_len + dec_seq_len*dec_seq_len + enc_seq_len*dec_seq_len)) * (3*layers + act_ckpt_layers) + 6*gbs*dec_seq_len*hs*vocab) / time_per_step
        hw_flops_per_gpu = hw_flops / (nodes * gpus_per_node)
        hw_tflops = hw_flops / 1e12
        hw_tflops_per_gpu = hw_flops_per_gpu / 1e12
    else:
        raise NotImplementedError("Model type not supported.")
    return round(model_tflops, 2), round(model_tflops_per_gpu, 2), round(hw_tflops, 2), round(hw_tflops_per_gpu, 2)


if __name__ == "__main__":
    main()
