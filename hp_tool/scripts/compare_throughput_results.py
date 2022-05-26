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

    training_logs = os.path.join(settings_cfg.get("logs"), "training_logs")
    candidate_configs = os.path.join(settings_cfg.get("logs"), "candidate_configs")
    final_result_logs = os.path.join(settings_cfg.get("logs"), "final_result")

    columns = ["Model Name", "Model Size", "TP", "PP", "MBS", "Act Ckpt Layers", "Num Layers", "Hidden Size", "GBS", "Nodes", "GPUs per Node", "Time per Step", "Samples per Second", "Model TFLOPS / GPU", "Model TFLOPS Aggregate", "HW TFLOPS / GPU", "HW TFLOPS Aggregate", "Full Config Name"]
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
        enc_seq_len = model_cfg.get("encoder_seq_length") if model_name == "gpt3" else model_cfg.get("seq_length")
        dec_seq_len = None if model_name == "gpt3" else data_cfg.get("seq_length_dec")
        hs = model_cfg.get("hidden_size")
        layers = model_cfg.get("num_layers")
        act_ckpt_layers = model_cfg.get("activations_checkpoint_num_layers")
        vocab = settings_cfg.get("vocab_size")
        nodes = trainer_cfg.get("num_nodes")
        gpus_per_node = trainer_cfg.get("devices")

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
                    half_timing_list = timing_list[len(timing_list) // 2 :]
                    avg_global_step_time = calculate_average(half_timing_list)
                    samples_per_s = round(gbs / avg_global_step_time, 2)
                    m_tflops, m_tflops_gpu, hw_tflops, hw_tflops_gpu = calculate_tflops
                    (
                            model_name=model_name,
                            gbs=gbs,
                            seq_len=enc_seq_len,
                            hs=hs,
                            layers=layers,
                            act_ckpt_layers=act_ckpt_layers,
                            vocab=vocab,
                            nodes=nodes,
                            gpus_per_node=gpus_per_node,
                            time_per_step=avg_global_step_time
                    )
                    result.append([model_name, model_size, tp, pp, mbs, act_ckpt_layers, layers, hs, gbs, nodes, gpus_per_node, avg_global_step_time, samples_per_s, m_tflops_gpu, model_tflops, hw_tflops_gpu, hw_tflops, candidate_dir])
                finally:
                    continue

    result.sort(key=lambda x: x[-1])
    print(f"Top {min(output_top_n, len(result)} configs sorted from fastest to slowest:")
    for i, (config, avg_time) in enumerate(result):
        print(f"Config #{i+1}: {config} with {avg_time:.4f}s per global step.")
        if i+1 == output_top_n:
            break
    print("\n==================================================")
    print(f"Optimal config: {result[0][-1]} with {result[0][11]:.4f}s per global step.")
    print(f"Saving config to {final_result_logs}/optimal_config_{model_size}b.yaml.")
    print("==================================================\n")

    # Save results as a CSV file.
    os.makedirs(final_result_logs, exist_ok=True)
    df = pd.DataFrame(result, columns=columns)
    df.to_csv(os.path.join(final_result_logs, "final_summary.csv"))

    copyfile(
        os.path.join(candidate_configs, f"{result[0][-1]}.yaml"),
        os.path.join(final_result_logs, f"optimal_config_{model_size}b.yaml"),
    )


def calculate_tflops(model_name, gbs, seq_len, hs, layers, act_ckpt_layers, vocab, nodes, gpus_per_node, time_per_step):
    """Calculates model and hardware TFLOPS for each model.

    Formulas:
    Model FLOPs = (24𝐵𝑠ℎ^2 + 4𝐵𝑠^2ℎ) x (3 x num_layers) + 6𝐵𝑠ℎ
    HW FLOPs = (24𝐵𝑠ℎ^2 + 4𝐵𝑠^2ℎ) x (3 x num_layers + num_checkpt_layers) + 6𝐵𝑠ℎ

    =round(((24*N3*2048*(M3^2) + 4*N3*(2048^2)*M3) *(3*L3) + (6*N3*2048*M3*50304)) / (G3*H3) / F3 / 1000000000000, 2)
    """
    if model_name == "gpt3":
        model_flops = (24*gbs*seq_len*hs*hs + 4*gbs*seq_len*seq_len*hs) * (3*layers) + (6*gbs*seq_len*hs*vocab) / time_per_step
        model_flops_per_gpu = model_flops / (nodes*gpus_per_node)
        model_tflops = model_flops / 1e12
        model_tflops_per_gpu = model_flops_per_gpu / 1e12
        hw_flops = (24*gbs*seq_len*hs*hs + 4*gbs*seq_len*seq_len*hs) * (3*layers + act_ckpt_layers) + (6*gbs*seq_len*hs*vocab) / time_per_step
        hw_flops_per_gpu = hw_flops / (nodes*gpus_per_node)
        hw_tflops = hw_flops / 1e12
        hw_tflops_per_gpu = hw_flops_per_gpu / 1e12
    elif model_name in ["t5", "mt5"]:
        pass
    else:
        raise NotImplementedError("Model type not supported.")
    return model_tflops, model_tflops_per_gpu, hw_tflops, hw_tflops_per_gpu

def calculate_average(ea_timing):
    time_avg = 0.0
    length = len(ea_timing)
    for step_time in ea_timing:
        time_avg += step_time.value
    return time_avg / length


if __name__ == "__main__":
    main()
