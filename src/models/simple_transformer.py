import torch
from torch import nn
from torch.nn import functional as F

class SimpleTransformer(nn.Module):
    def __init__(self, num_classes: int, n_head: int, d_model: int, d_ff: int, num_layers: int, dropout: float = 0.1):
        """
        Simple Transformer model for audio classification.
        Args:
            num_classes (int): Number of output classes.
            n_head (int): Number of attention heads.
            d_model (int): Dimension of the model.
            dim_ff (int): Dimension of the feedforward network.
            num_layers (int): Number of transformer layers.
        """

        super().__init__()

        self.num_classes = num_classes
        self.n_head = n_head
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout


        self.embedding = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(3, self.d_model // 8, kernel_size=3, stride=2), # 116x116x3 -> 58x58xd_model/8
            nn.ReLU(inplace=True),
            nn.Conv2d(self.d_model // 8, self.d_model // 4, kernel_size=3, stride=2), # 58x58xd_model/8 -> 29x29xd_model/4
            nn.ReLU(inplace=True),
            nn.Conv2d(self.d_model // 4, self.d_model // 2, kernel_size=3, stride=2), # 29x29xd_model/4 -> 14x14xd_model/2
            nn.ReLU(inplace=True),
            nn.Conv2d(self.d_model // 2, self.d_model, kernel_size=3, stride=2), # 14x14xd_model/2 -> 7x7xd_model
            nn.BatchNorm2d(self.d_model),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)), # 7x7xd_model -> 1x1xd_model
            nn.Flatten(), # 1x1xd_model -> d_model
        )

        # Currenyly not implementing positional encoding, because sequence length is 1.
        self.positional_encoding = None
        self.transformer_blocks = nn.Sequential(
            *[
                nn.TransformerEncoderLayer(
                    d_model=self.d_model,
                    nhead=self.n_head,
                    dim_feedforward=self.d_ff,
                    activation="relu",
                    dropout=self.dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.classifier = nn.Linear(self.d_model, self.num_classes)


    def forward(self, x):
        """
        Forward pass of the model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 128).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """

        x = self.embedding(x)  # Shape: (batch_size, d_model)
        if self.positional_encoding:
            x = self.positional_encoding(x)  # Add positional encoding if implemented

        x = x.unsqueeze(0)  # Add sequence dimension: (1, batch_size, d_model)
        x = self.transformer_blocks(x)  # Shape: (1, batch_size, d_model)
        x = x.squeeze(0)  # Remove sequence dimension: (batch_size, d_model)
        x = self.classifier(x)  # Shape: (batch_size, num_classes)

        return x


if __name__ == "__main__":
    model = SimpleTransformer(12, 8, 512, 2048, 6).cuda()
    x = torch.randn(256, 3, 128, 128).cuda()  # Example input
    output = model(x)
    print(output.shape)  # Should be (256, num_classes)