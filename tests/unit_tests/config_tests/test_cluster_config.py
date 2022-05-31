from omegaconf import OmegaConf


class TestClusterConfig:
    def test_cluster_bcm_config(self):
        conf = OmegaConf.load("conf/cluster/bcm.yaml")
        s = """
        partition: null
        account: null
        exclusive: True
        gpus_per_task: null
        gpus_per_node: 8
        mem: 0
        overcommit: False
        job_name_prefix: "bignlp-"
        """
        expected = OmegaConf.create(s)
        assert (
            expected == conf
        ), f"conf/cluster/bcm.yaml must be set to {expected} but it currently is {conf}."
