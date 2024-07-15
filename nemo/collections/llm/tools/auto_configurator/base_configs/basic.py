class Basic:
    def __init__(
        self,
        name: str = None,
        version: int = None,
        size: int = None,
        cfg: dict = {},
    ):
        self.name = name
        self.version = version
        self.size = size
        self.num_nodes = cfg.get("num_nodes", 8)
        self.num_gpus = cfg.get("num_gpus", 8)
        self.max_steps = cfg.get("max_steps", 50)
        self.seq_length = cfg.get("seg_length", 2048)
        self.global_batch_size = cfg.get("global_batch_size", 2048)

    def model_config(self):
        None

    def optim_config(self):
        None

    def tokenizer_config(self):
        None

    def trainer_config(self):
        None

    def data_config(self):
        None
