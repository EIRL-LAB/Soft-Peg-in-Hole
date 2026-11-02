import torch
import torch.nn.functional as F
from torch.nn import Conv2d, Module

import resnet


class FCN(Module):
    """ResNet-18 FCN with a dense center heatmap head."""

    def __init__(
        self,
        num_input_channels=1,
        num_output_channels=1,
    ):
        super().__init__()

        self.resnet18 = resnet.resnet18(
            num_input_channels=num_input_channels,
        )
        self.conv1 = Conv2d(512, 128, kernel_size=1)
        self.conv2 = Conv2d(128, 32, kernel_size=1)
        self.conv3 = Conv2d(32, num_output_channels, kernel_size=1)
        if self.conv3.bias is not None:
            torch.nn.init.constant_(self.conv3.bias, -2.19)

    def forward(self, x):
        x = self.resnet18.features(x)
        x = F.relu(self.conv1(x), inplace=True)
        x = F.interpolate(
            x,
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )
        x = F.relu(self.conv2(x), inplace=True)
        x = F.interpolate(
            x,
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )
        return self.conv3(x)


if __name__ == "__main__":
    model = FCN()
    num_params = sum(param.numel() for param in model.parameters())
    print("Total: %.2fM" % (num_params / 1e6))
    image = torch.ones((1, 1, 240, 320))
    output = model(image)
    print(output.size())
