import os
import sys
import json
from CITestHelper import CITestHelper
import time
from os.path import exists
import yaml

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
    if model_name == "gpt3":
        act_term = (
            4 * gbs * enc_seq_len * enc_seq_len * hs * layers
            if act_ckpt_layers == "selective"
            else 0
        )

        # Model FLOPS calculation
        model_flops = (
            (24 * gbs * enc_seq_len * hs * hs + 4 * gbs * enc_seq_len * enc_seq_len * hs)
            * (3 * layers)
            + (6 * gbs * enc_seq_len * hs * vocab)
        ) / time_per_step
        model_flops_per_gpu = model_flops / (nodes * gpus_per_node)
        model_tflops = model_flops / 1e12
        model_tflops_per_gpu = model_flops_per_gpu / 1e12
        # HW FLOPS calculation
        if act_ckpt_layers == "selective":
            hw_flops = (
                (24 * gbs * enc_seq_len * hs * hs + 4 * gbs * enc_seq_len * enc_seq_len * hs)
                * (3 * layers)
                + act_term
                + (6 * gbs * enc_seq_len * hs * vocab)
            ) / time_per_step
        else:
            hw_flops = (
                (24 * gbs * enc_seq_len * hs * hs + 4 * gbs * enc_seq_len * enc_seq_len * hs)
                * (3 * layers + act_ckpt_layers)
                + (6 * gbs * enc_seq_len * hs * vocab)
            ) / time_per_step
        hw_flops_per_gpu = hw_flops / (nodes * gpus_per_node)
        hw_tflops = hw_flops / 1e12
        hw_tflops_per_gpu = hw_flops_per_gpu / 1e12
    elif model_name in ["t5", "mt5"]:
        # Model FLOPS calculation
        model_flops = (
            (
                2 * gbs * hs * hs * (5 * enc_seq_len + 4 * dec_seq_len)
                + 6 * gbs * hs * ffn_hs * (enc_seq_len + dec_seq_len)
                + 4
                * gbs
                * hs
                * (
                    enc_seq_len * enc_seq_len
                    + dec_seq_len * dec_seq_len
                    + enc_seq_len * dec_seq_len
                )
            )
            * 3
            * layers
            + 6 * gbs * dec_seq_len * hs * vocab
        ) / time_per_step
        model_flops_per_gpu = model_flops / (nodes * gpus_per_node)
        model_tflops = model_flops / 1e12
        model_tflops_per_gpu = model_flops_per_gpu / 1e12
        # HW FLOPS calculation
        hw_flops = (
            (
                2 * gbs * hs * hs * (5 * enc_seq_len + 4 * dec_seq_len)
                + 6 * gbs * hs * ffn_hs * (enc_seq_len + dec_seq_len)
                + 4
                * gbs
                * hs
                * (
                    enc_seq_len * enc_seq_len
                    + dec_seq_len * dec_seq_len
                    + enc_seq_len * dec_seq_len
                )
            )
            * (3 * layers + act_ckpt_layers)
            + 6 * gbs * dec_seq_len * hs * vocab
        ) / time_per_step
        hw_flops_per_gpu = hw_flops / (nodes * gpus_per_node)
        hw_tflops = hw_flops / 1e12
        hw_tflops_per_gpu = hw_flops_per_gpu / 1e12
    else:
        raise NotImplementedError("Model type not supported.")
    return (
        round(model_tflops, 2),
        round(model_tflops_per_gpu, 2),
        round(hw_tflops, 2),
        round(hw_tflops_per_gpu, 2),
    )

if __name__ == '__main__':
    args = sys.argv[1:]
    ci_job_results_dir = args[0] # eg. '/home/shanmugamr/bignlp-scripts/results/train_gpt3_126m_tp1_pp1_1node_100steps'
    git_results_dir_path = args[1]
    base_image = args[2]
    nodes = int(args[3])
    current_timestamp = time.time()

    train_time_list = CITestHelper.read_tb_logs_as_list(ci_job_results_dir, "train_step_timing")
    train_time_list = train_time_list[len(train_time_list) // 2:]  # Discard the first half.
    train_time_avg = sum(train_time_list) / len(train_time_list)
    model_name = ci_job_results_dir.rsplit("/", 1)[1].split("_")[0]
    model_size = ci_job_results_dir.rsplit("/", 1)[1].split("_")[1]
    if model_name == "gpt3":
        enc_seq_len = 2048
        dec_seq_len = None
        vocab = 51200
    elif model_name == "mt5":
        enc_seq_len = 512
        dec_seq_len = 128
        vcoab = 250000
    elif model_name == "t5":
        enc_seq_len = 512
        dec_seq_len = 128
        vocab = 29000
    yaml_file = os.path.join("conf/training", model_name, model_size + ".yaml")
    yaml_contents = yaml.safe_load(open(yaml_file, 'r'))
    gbs = yaml_contents['model']['global_batch_size']
    if model_name == "gpt3":
        hs = yaml_contents['model']['hidden_size'] 
        layers = yaml_contents['model']['num_layers']
    else :
        hs = yaml_contents['model']['encoder']['hidden_size']
        layers = yaml_contents['model']['encoder']['num_layers']   
    ffn_hs = hs * 4 
    tflops = calculate_tflops(
            model_name,
            gbs,
            enc_seq_len,
            dec_seq_len,
            hs,
            ffn_hs,
            layers,
            0,
            vocab,
            nodes,
            8,
            train_time_avg,
        )[1]
    output_file_name = ci_job_results_dir.rsplit("/", 1)[1]+".json"
    release_perf_file = os.path.join(git_results_dir_path, output_file_name)

    if not exists(release_perf_file):
        new_result={}
        new_result[base_image] = [current_timestamp, train_time_avg, tflops]
        final_result={}
        final_result["train_time_metrics"] = new_result
        final_result["peak_memory_metrics"] = {}
        with open(release_perf_file, "w") as f:
            json.dump(final_result, f)
    else:
        with open(release_perf_file) as f:
            previous_results = json.load(f)    
        previous_results["train_time_metrics"][base_image] = [current_timestamp, train_time_avg, tflops] 
        final_result = previous_results
        with open(release_perf_file, 'w') as f:
             json.dump(final_result, f)

    print(f" ****** Release Performance timeline : {final_result} logged in  {release_perf_file}", flush=True)
