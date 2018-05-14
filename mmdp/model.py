import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn1 = nn.Conv2d(1, 1, 3)
        self.cnn2 = nn.Conv2d(1, 1, 3)
        self.cnn3 = nn.Conv2d(1, 1, 2)

    def forward(self, x):
        out = self.cnn1(x)
        out = self.cnn2(out)
        out = self.cnn3(out)
        return out

