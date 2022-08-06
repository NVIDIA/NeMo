import os
import sys
import json
from CITestHelper import CITestHelper
import time

if __name__ == '__main__':
    args = sys.argv[1:]
    ci_job_results_dir = args[0] # eg. '/home/shanmugamr/bignlp-scripts/results/train_gpt3_126m_tp1_pp1_1node_100steps'
    git_results_dir_path = args[1]

    train_time_list = CITestHelper.read_tb_logs_as_list(ci_job_results_dir, "train_step_timing")
    train_time_list = train_time_list[len(train_time_list) // 2:]  # Discard the first half.
    train_time_avg = sum(train_time_list) / len(train_time_list)

    train_metrics = {"train_step_timing_avg": train_time_avg}
    output_file_name = ci_job_results_dir.rsplit("/", 1)[1]+".json"
    train_time_file = os.path.join(git_results_dir_path, output_file_name)

    with open(train_time_file) as f:
        previous_results = json.load(f)

    previous_results['best'] = previous_results[min(previous_results, key=previous_results.get)] # Will update the best value in the results
    release_version = "22.06"
    current_timestamp = time.time()
    previous_results[release_version + "-" current_timestamp] = train_time_avg # Will put in the latest result
    with open(train_time_file, 'w') as f:
         f.write(json.dumps(previous_results))

    print(f" ****** Release Performance timeline : {train_metrics} logged in  {train_time_file}", flush=True)
