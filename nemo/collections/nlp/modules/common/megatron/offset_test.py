def get_offset(parallelization_specs, pipeline_model_rank):
    offset = 0
    # pipeline_rank = parallel_state.get_pipeline_component_parallel_rank()
    # pipeline_model_rank = 2
    pipeline_model_rank_tracker = 0
    component_tracker = 0
    while pipeline_model_rank > pipeline_model_rank_tracker:
        component_name = list(parallelization_specs.keys())[component_tracker]
        component_num_layers = len(parallelization_specs[component_name]['layers'])
        pipeline_component_parallel_group_size = parallelization_specs[component_name]['pipeline_model_parallel_group_size']
        assert (
                    component_num_layers % pipeline_component_parallel_group_size == 0
                ), 'component_num_layers must be divisible by pipeline_component_parallel_group_size'
        for pipeline_component_parallel_group_rank in range(pipeline_component_parallel_group_size):
            if pipeline_model_rank_tracker < pipeline_model_rank:
                offset += component_num_layers // pipeline_component_parallel_group_size
            pipeline_model_rank_tracker += 1


        component_tracker += 1

    
    print('offset: ' + str(offset))


if __name__ == "__main__":
    parallelization_specs = {
        "stimulus": {
            "layers": [0],
            "gpu_ranks": [0,1,2,3,4,5,6,7],
            "gpus_per_node": 1,
            "data_parallel_group_size": 1,
            "tensor_model_parallel_group_size": 8,
            "pipeline_model_parallel_group_size": 1,
            "micro_batch_size": 2
        },
        "test": {
            "layers": [1],
            "gpu_ranks": [8,9,10,11,12,13,14,15],
            "gpus_per_node": 8,
            "data_parallel_group_size": 1,
            "tensor_model_parallel_group_size": 8,
            "pipeline_model_parallel_group_size": 1,
            "micro_batch_size": 2
        },
        "response": {
            "layers": [2],
            "gpu_ranks": [16,17,18,19,20,21,22,23],
            "gpus_per_node": 8,
            "data_parallel_group_size": 1,
            "tensor_model_parallel_group_size": 8,
            "pipeline_model_parallel_group_size": 1,
            "micro_batch_size": 2
        }
    }
    for i in range(3):
        get_offset(parallelization_specs, i)