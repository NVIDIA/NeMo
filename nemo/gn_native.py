from torch import nn


class GroupNormNormlization(nn.GroupNorm):
    def __init__(self, *args, act="", **kwargs):
        super().__init__(*args, **kwargs)
        if act == "silu":
            self.act = nn.SiLU()
        elif act == "relu":
            self.act = nn.ReLU()
        elif act == "":
            self.act = None
        else:
            raise ValueError(f"Unknown activation {act}")

    def forward(self, x):
        x = super().forward(x)
        if self.act is None:
            return x
        else:
            return self.act(x)
