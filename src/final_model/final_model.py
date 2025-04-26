import sys

sys.path.append(".\src")


import torch
from torch import nn
from torchvision.models import ResNet

from models.simple_transformer import SimpleTransformer
from speech_activation.cnn_sa import CNN_SA

class FinalModel(nn.Module):
    def __init__(self, base_10: SimpleTransformer, known_unknown: ResNet, voice_silence: CNN_SA):
        """
        Final model that combines three sub-models: base_10, known_unknown, and voice_silence.

        Args:
            base_10 (SimpleTransformer): The base transformer model for 10 classes.
            known_unknown (ResNet): The ResNet model for known/unknown classification.
            voice_silence (CNN_SA): The CNN model for voice/silence classification.
        """
        super(FinalModel, self).__init__()
        self.base_10 = base_10
        self.known_unknown = known_unknown
        self.voice_silence = voice_silence

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 116, 116).

        Returns:
            tuple: A tuple containing the outputs from the three sub-models.
        """
        x1 = self.voice_silence(x)
        x2 = self.known_unknown(x)
        x3 = self.base_10(x)

        return x1, x2, x3
