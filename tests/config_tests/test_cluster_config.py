from omegaconf import OmegaConf


class TestClusterConfig:
    
    def test_cluster_bcm_config(self):
        conf = OmegaConf.load('conf/cluster/bcm.yaml')
        s = """
        partition: null
        account: null
        exclusive: True
        gpus_per_task: 1
        mem: 0
        overcommit: True
        job_name_prefix: "bignlp_hp_tool:"

        cluster:
          type: pyxis
          account: null
          partition: null
          srun_args: ["--mpi", "pmix"]
          support_gpus_allocation: False
        env:
          job_name_prefix: "bignlp_hp_tool:"
          training_container_image: ${training_container}
          inference_container_image: ${inference_container}
        """
        expected = OmegaConf.create(s)
        assert expected == conf, f"conf/cluster/bcm.yaml must be set to {expected} but it currently is {conf}."

