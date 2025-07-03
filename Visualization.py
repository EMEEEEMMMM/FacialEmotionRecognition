import matplotlib.pyplot as plt
import visualtorch
from torch import nn
import torch


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            # Input 3x48x48
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # Input 64x24x24
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # Input 128x12x12
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # Input 256x6x6
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 8),
        )

    def forward(self, x):
        return self.cnn(x)

model = CNN()

tensor = (1, 3, 48, 48)

img = visualtorch.layered_view(model, input_shape=tensor, legend=True)
img1 = visualtorch.graph_view(model, input_shape=tensor,node_size=20,node_spacing=0)


plt.tight_layout()
plt.figure(figsize=(1.8,1),dpi=1000)
plt.axis("off")
plt.imshow(img)
plt.savefig("img",bbox_inches="tight", pad_inches=0)
plt.imshow(img1)
plt.savefig("img2", bbox_inches="tight", pad_inches=0)
