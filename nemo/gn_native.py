from torch import nn


class GroupNormNormlization(nn.GroupNorm):
    def __init__(self, *args, act="", **kwargs):
        super().__init__(*args, **kwargs)
        if act == "silu":
            self.act = nn.SiLU()
        elif act == "relu":
            self.act = nn.ReLU()
        elif act == "":
            self.act = nn.Identity()
        else:
            raise ValueError(f"Unknown activation {act}")

    def forward(self, x):
        x = super().forward(x)
        return self.act(x)
