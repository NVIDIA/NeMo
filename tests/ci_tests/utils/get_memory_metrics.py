import os
import sys
import json
import time
import numpy
from os.path import exists

if __name__ == '__main__':
    args = sys.argv[1:]
    ci_job_results_dir = args[0] # eg. '/home/shanmugamr/bignlp-scripts/results/train_gpt3_126m_tp1_pp1_1node_100steps'
    git_results_dir_path = args[1]
    base_image = args[2]
    current_timestamp = time.time()
    memory_metrics_log_file = list(filter(lambda x: "mem" in x,os.listdir(ci_job_results_dir)))[0] # output file of name log-joc-bignlp_ci:gpt3_126m_tp1_pp1_1nodes_bf16_precision_O2_10steps_mem_2948081.out
    memory_metrics_log_file = os.path.join(ci_job_results_dir, memory_metrics_log_file)
    
    max_memory_used = max(numpy.genfromtxt(memory_metrics_log_file, delimiter=' ', skip_header=2)[:,-2]) 
    output_file_name = ci_job_results_dir.rsplit("/", 1)[1]+".json"
    release_perf_file = os.path.join(git_results_dir_path, output_file_name)

    if not exists(release_perf_file):
        new_result={}
        new_result[base_image] = [current_timestamp,max_memory_used]
        final_result={}
        final_result["train_time_metrics"] = {}
        final_result["peak_memory_metrics"] = new_result
        with open(release_perf_file, "w") as f:
            json.dump(final_result, f)
    else:
        with open(release_perf_file) as f:
            previous_results = json.load(f)    
        previous_results["peak_memory_metrics"][base_image] = [current_timestamp,max_memory_used]
        final_result = previous_results
        with open(release_perf_file, 'w') as f:
             json.dump(final_result, f)

    print(f" ****** Release Performance timeline : {final_result} logged in  {release_perf_file}", flush=True)
