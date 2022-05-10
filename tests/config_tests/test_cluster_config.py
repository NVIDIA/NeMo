from omegaconf import OmegaConf


class TestClusterConfig:
    def test_cluster_bcm_config(self):
        conf = OmegaConf.load("conf/cluster/bcm.yaml")
        s = """
        cluster:
          type: pyxis
          account: null
          partition: null
          srun_args: ["--mpi", "pmix"]
          support_gpus_allocation: True
        env:
          job_name_prefix: "bignlp_hp_tool:"
          training_container_image: nvcr.io/ea-bignlp/bignlp-training:22.04.01-py3
          inference_container_image: nvcr.io/ea-bignlp/bignlp-inference:22.04.01-py3
        
        exclusive: True
        gpus_per_task: 1
        gpus_per_node: null
        mem: 0
        overcommit: True
        
        partition: ${cluster.cluster.partition}
        account: ${cluster.cluster.account}
        job_name_prefix: ${cluster.env.job_name_prefix}
        """
        expected = OmegaConf.create(s)
        assert (
            expected == conf
        ), f"conf/cluster/bcm.yaml must be set to {expected} but it currently is {conf}."
