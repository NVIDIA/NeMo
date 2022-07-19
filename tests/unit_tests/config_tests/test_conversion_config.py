from omegaconf import OmegaConf


class TestConversionT5Config:
    def test_conversion_t5_config(self):
        conf = OmegaConf.load("conf/conversion/convert_t5.yaml")
        s = """
        run:
          job_name: convert_${conversion.run.model_train_name}
          nodes: ${divide_ceil:${conversion.model.model_parallel_size}, 8} # 8 gpus per node
          time_limit: "2:00:00"
          ntasks_per_node: ${divide_ceil:${conversion.model.model_parallel_size}, ${.nodes}}
          convert_name: convert_nemo
          model_train_name: t5_220m
          train_dir: ${base_results_dir}/${.model_train_name}
          results_dir: ${.train_dir}/${.convert_name}
          output_path: ${.train_dir}/${.convert_name}
          nemo_file_name: megatron_t5.nemo # name of nemo checkpoint; must be .nemo file
        
        model:
          model_type: t5 # gpt or t5, use t5 for mt5 as well
          checkpoint_folder: ${conversion.run.train_dir}/checkpoints
          checkpoint_name: latest # latest OR name pattern of a checkpoint (e.g. megatron_gpt-*last.ckpt)
          hparams_file: ${conversion.run.train_dir}/hparams.yaml
          tensor_model_parallel_size: 1 # 1 for 220m, 2 for 3b
          pipeline_model_parallel_size: 1
          model_parallel_size: ${multiply:${.tensor_model_parallel_size}, ${.pipeline_model_parallel_size}}
          vocab_file: ${data_dir}/bpe/vocab.txt
          merge_file: null
        """
        expected = OmegaConf.create(s)
        assert (
            expected == conf
        ), f"conf/conversion/convert_t5.yaml must be set to {expected} but it currently is {conf}."


class TestConversionGPT3Config:
    def test_conversion_gpt3_config(self):
        conf = OmegaConf.load("conf/conversion/convert_gpt3.yaml")
        s = """
        run:
          job_name: convert_${conversion.run.model_train_name}
          nodes: ${divide_ceil:${conversion.model.model_parallel_size}, 8}
          time_limit: "2:00:00"
          ntasks_per_node: ${divide_ceil:${conversion.model.model_parallel_size}, ${.nodes}}
          convert_name: convert_nemo
          model_train_name: gpt3_5b
          train_dir: ${base_results_dir}/${.model_train_name}
          results_dir: ${.train_dir}/${.convert_name}
          output_path: ${.train_dir}/${.convert_name}
          nemo_file_name: megatron_gpt.nemo

        model:
          model_type: gpt
          checkpoint_folder: ${conversion.run.train_dir}/checkpoints
          checkpoint_name: latest
          hparams_file: ${conversion.run.train_dir}/hparams.yaml
          tensor_model_parallel_size: 2
          pipeline_model_parallel_size: 1
          model_parallel_size: ${multiply:${.tensor_model_parallel_size}, ${.pipeline_model_parallel_size}}
          vocab_file: ${data_dir}/bpe/vocab.json
          merge_file: ${data_dir}/bpe/merges.txt
        """
        expected = OmegaConf.create(s)
        assert (
            expected == conf
        ), f"conf/conversion/convert_gpt3.yaml must be set to {expected} but it currently is {conf}."
