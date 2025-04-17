import torch
from torch import nn
from torch.nn import functional as F

class SimpleTransformer(nn.Module):
    def __init__(self, num_classes: int = 12):
        """
        Simple Transformer model for audio classification.
        Args:
            num_classes (int): Number of classes for classification.
        """

        super().__init__()
        # TODO
        self.lin1 = nn.Linear(128*128, 256)
        self.lin2 = nn.Linear(256, 128)
        self.classifier = nn.LazyLinear(num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass of the model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 128).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """
        x = nn.Flatten()(x)
        x = self.relu(self.lin1(x))
        x = self.relu(self.lin2(x))
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    model = SimpleTransformer().cuda()
    x = torch.randn(256, 128, 128).cuda()  # Example input
    output = model(x)
    print(output.shape)  # Should be (256, num_classes)