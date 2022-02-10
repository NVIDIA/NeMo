from omegaconf import OmegaConf


class TestConfig:

    def test_config(self):
        conf = OmegaConf.load('conf/config.yaml')
        s = """
        defaults:
          - _self_
          - cluster: bcm
          - data_preparation: download_pile
          - training: 5b
          - conversion: convert
          - evaluation: evaluate_all
          - override hydra/job_logging: stdout

        hydra:
          run:
            dir: .
          output_subdir: null

        run_data_preparation: True
        run_training: True
        run_conversion: True
        run_evaluation: True

        cluster_type: bcm
        training_config: 5b
        bignlp_path: ???
        data_dir: ${bignlp_path}/data
        base_results_dir: ${bignlp_path}/results
        container_mounts:
          - null
        container: nvcr.io/ea-bignlp/bignlp-training:22.01-py3

        wandb_api_key_file: null
        nccl_topology_xml_file: null
        """
        expected = OmegaConf.create(s)
        assert expected == conf, f"conf/config.yaml must be set to {expected} but it currently is {conf}."
