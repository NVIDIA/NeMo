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

        cluster_type: bcm  # bcm or bcp
        auto_configurator_path: ???  # Path to the location of auto_configurator codebase.
        launcher_scripts_path: ${auto_configurator_path}/../launcher_scripts
        base_results_dir: ${auto_configurator_path}/results
        data_dir: ${launcher_scripts_path}/data

        training_container: nvcr.io/nvidia/nemo:24.05
        container_mounts:
          - null
        
        wandb:
          enable: False
          api_key_file: null
          project: nemo-megatron-autoconfig

        search_config_value: ${hydra:runtime.choices.search_config}
        """
        expected = OmegaConf.create(s)
        assert expected == conf, f"conf/config.yaml must be set to {expected} but it currently is {conf}."
