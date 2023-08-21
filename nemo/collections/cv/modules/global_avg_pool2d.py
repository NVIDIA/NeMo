from nemo.core.classes.module import NeuralModule

class GlobalAvgPool2d(NeuralModule):

    def __init__(self, dim=(2, 3), keepdim=False):
        super(GlobalAvgPool2d, self).__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x):

        return  x.mean(dim=self.dim, keepdim=self.keepdim)