from omegaconf import OmegaConf


class TestConfig:
    def test_config(self):
        conf = OmegaConf.load("conf/config.yaml")
        s = """
        defaults:
          - _self_
          - cluster: bcm
          - search_config: gpt3/5b
          - override hydra/job_logging: stdout

        hydra:
          run:
            dir: .
          output_subdir: null

        run_training_hp_search: True
        run_inference_hp_search: True

        bignlp_hp_tool_path: ???
        bignlp_scripts_path: ${bignlp_hp_tool_path}/../bignlp-scripts
        base_results_dir: ${bignlp_hp_tool_path}/results
        data_dir: ${bignlp_scripts_path}/data

        training_container: nvcr.io/ea-bignlp/bignlp-training:22.08-py3
        
        wandb:
          enable: False
          api_key_file: null
          project: bignlp-hp-tool

        search_config_value: ${hydra:runtime.choices.search_config}
        """
        expected = OmegaConf.create(s)
        assert (
            expected == conf
        ), f"conf/config.yaml must be set to {expected} but it currently is {conf}."
