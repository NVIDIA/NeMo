from nemo.core.classes import NeuralModule


class Permute(NeuralModule):
    def __init__(self, dims, make_contiguous=False):
        super(Permute, self).__init__()
        self.dims = dims
        self.make_contiguous = make_contiguous

    def forward(self, x):
        x = x.permute(self.dims)
        if self.make_contiguous:
            x = x.contiguous()
        return x
