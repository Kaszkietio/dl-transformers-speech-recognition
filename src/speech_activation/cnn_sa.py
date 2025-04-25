import torch
from torch import nn
from torch.nn import functional as F


class CNN_SA(nn.Module):
    def __init__(
            self,
            num_classes: int,
            dim_model: int,
            num_heads: int,
            num_layers: int,
            dim_ff: int,
            dropout: float = 0.1
        ):
        """
        CNN-based model for audio classification.
        Args:
            num_classes (int): Number of output classes.
            dim_model (int): Dimension of the model.
            num_heads (int): Number of attention heads.
            num_layers (int): Number of transformer layers.
            dim_ff (int): Dimension of the feedforward network.
            dropout (float): Dropout rate.
        """

        super().__init__()

        self.num_classes = num_classes
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dim_ff = dim_ff
        self.dropout = dropout


        self.embedding = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1), # 3x116x116 -> 16x116x116
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((3, 1)), # 16x116x116 -> 16x38x116
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), # 16x38x116 -> 32x38x116
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)), # 32x38x116 -> 32x19x116
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1), # 32x19x116 -> 32x19x116
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)), # 32x19x116 -> 32x9x116
        )
        self.final_embedding = nn.Linear(288, self.dim_model) # 288 -> dim_model

        self.transformer_blocks = nn.Sequential(
            *[
                nn.TransformerEncoderLayer(
                    d_model=self.dim_model,
                    nhead=self.num_heads,
                    dim_feedforward=self.dim_ff,
                    activation="relu",
                    dropout=self.dropout,
                    batch_first=True
                )
                for _ in range(self.num_layers)
            ]
        )

        self.classifier = nn.Linear(self.dim_model, self.num_classes)


    def forward(self, x: torch.Tensor):
        """
        Forward pass of the model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 116, 116).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """
        x = self.embedding(x) # Bx3x116x116 -> Bx32x9x116
        x = torch.flatten(x, start_dim=1, end_dim=2) # Bx32x9x116 -> Bx288x116
        x = x.transpose(-2, -1) # Bx288x116 -> Bx116x288
        x = self.final_embedding(x) # Bx116x288 -> Bx116xd_model
        x = self.transformer_blocks(x) # Bx116xd_model -> Bx116xd_model
        x = self.classifier(x.mean(dim=1)) # Bx116xd_model -> Bxd_model -> Bxnum_classes
        return x


if __name__ == "__main__":
    torch.manual_seed(0)
    model = CNN_SA(num_classes=2, dim_model=256, num_heads=4, num_layers=1, dim_ff=2048, dropout=0.1).cuda()
    x = torch.randn(512, 3, 116, 116).cuda()  # Example input
    x = model(x)  # Forward pass
    print(x.shape)  # Output shape should be (512, 2)