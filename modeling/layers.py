from torch import nn


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], x.shape[1], -1)

    def __repr__(self):
        return self.__class__.__name__ + '()'