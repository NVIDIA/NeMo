import os
import sys
import json
from CITestHelper import CITestHelper

def collect_val_test_metrics(ci_job_results_dir):
    # TODO: Fetch current baseline

    # val loss
    val_loss_list = CITestHelper.read_tb_logs_as_list(ci_job_results_dir, "val_loss")

    val_metrics = {
        "val_loss": {
            "start_step": 0,
            "end_step": 1,
            "step_interval": 1,
            "values": val_loss_list[:1]
        }
    }

    output_file_name = ci_job_results_dir.rsplit("/",1)[1]+".json"
    val_metrics_file = os.path.join(ci_job_results_dir, output_file_name)
    with open(val_metrics_file, "w") as out_file:
        json.dump(val_metrics, out_file)
    print(f" ****** CI val metrics logged in {val_metrics_file}", flush=True)

def collect_train_test_metrics(ci_job_results_dir):
    # TODO: Fetch current baseline

    # train loss
    train_loss_list = CITestHelper.read_tb_logs_as_list(ci_job_results_dir, "reduced_train_loss")

    # val loss
    val_loss_list = CITestHelper.read_tb_logs_as_list(ci_job_results_dir, "val_loss")

    # step timing
    train_time_list = CITestHelper.read_tb_logs_as_list(ci_job_results_dir, "train_step_timing")
    train_time_list = train_time_list[len(train_time_list) // 2:]  # Discard the first half.
    train_time_avg = sum(train_time_list) / len(train_time_list)

    train_metrics = {
        "reduced_train_loss": {
            "start_step": 0,
            "end_step": 100,
            "step_interval": 5,
            "values": train_loss_list[0:100:5],
        },
        "val_loss": {
            "start_step": 0,
            "end_step": 5,
            "step_interval": 1,
            "values": val_loss_list[0:5],
        },
        "train_step_timing_avg": train_time_avg,
    }
    output_file_name = ci_job_results_dir.rsplit("/",1)[1]+".json"
    train_metrics_file = os.path.join(ci_job_results_dir, output_file_name)
    with open(train_metrics_file, "w") as out_file:
        json.dump(train_metrics, out_file)
    print(f" ****** CI train metrics logged in {train_metrics_file}", flush=True)
    str_train_metrics = str(train_metrics).replace("'", "\"")
    print(f" ****** CI train metrics: \n {str_train_metrics}", flush=True)

if __name__ == '__main__':
    args = sys.argv[1:]
    ci_job_results_dir = args[0] # eg. '/home/shanmugamr/bignlp-scripts/results/train_gpt3_126m_tp1_pp1_1node_100steps'
    run_stage = args[1] # eg. train
    run_model = args[2] # eg. gpt3

    if run_stage in ["train", "finetune", "prompt_learn"]:
        collect_train_test_metrics(ci_job_results_dir)

    if run_stage in ["eval"] and run_model in ["t5", "mt5"]:
        collect_val_test_metrics(ci_job_results_dir)
