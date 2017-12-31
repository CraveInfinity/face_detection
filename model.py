import torch
import torch.nn as nn


class Net(nn.Module):

    def __init__(self, batch_size):
        super(Net, self).__init__()
        self.batch_size = batch_size


        self.features_1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=5, padding=2),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(),
            nn.MaxPool2d(5),
        )

        self.features_2 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=5, padding=2),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(),
            nn.MaxPool2d(5),
        )

        self.full_connected = nn.Sequential(
            nn.Linear(16*6*6, 100),
            nn.LeakyReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(100, 1)
        )

    def forward(self, inputs):
        a = self.features_1(inputs[0])
        b = self.features_2(inputs[1])

        out = torch.cat([a, b], dim=1)
        out = out.view(self.batch_size, -1, 16*6*6)
        out = self.full_connected(out)
        out = torch.squeeze(out, dim=1)

        return out