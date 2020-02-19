import torch
import torch.nn as nn


class PlantDiseaseNet(nn.Module):
    def __init__(self, training: bool = False):
        super(PlantDiseaseNet, self).__init__()
        self.conv1 = self.nnlayer(3, 24)
        self.conv2 = self.nnlayer(24, 24)
        self.conv3 = self.nnlayer(24, 48)

        self.pool1 = self.pool(2)
        self.drop1 = self.dropout(0.2)

        self.conv4 = self.nnlayer(48, 48)
        self.conv5 = self.nnlayer(48, 48)
        self.conv6 = self.nnlayer(48, 48)

        self.pool2 = self.pool(2)
        self.drop2 = self.dropout(0.2)

        self.dense1 = nn.Linear(48 * 50 * 50, 1024)
        self.drop3 = self.dropout(0.3)
        self.dense2 = nn.Linear(1024, 15)
        self.drop4 = self.dropout(0.3)

    def nnlayer(self, chans_in: int, chans_out: int):
        return nn.Sequential(
            nn.Conv2d(chans_in, chans_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(chans_out, eps=1e-5, momentum=0.9),
            nn.ReLU(inplace=True),
        )

    def pool(self, k_size: int):
        return nn.MaxPool2d(
            kernel_size=k_size, stride=k_size, dilation=1, ceil_mode=False
        )

    def dropout(self, p: float):
        return nn.Dropout2d(p)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.pool1(out)
        out = self.drop1(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.pool2(out)
        out = self.drop2(out)
        out = out.view(out.size(0), -1)
        out = self.dense1(out)
        out = self.drop3(out)
        out = self.dense2(out)
        out = self.drop4(out)
        return out
