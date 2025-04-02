from nemo.common.plan.plan import Plan


class Missing(Plan):
    def __init__(self, name: str):
        super().__init__()
        self.name = name

    def execute(self):
        raise ValueError(f"`{self.name}` is required, got None")
