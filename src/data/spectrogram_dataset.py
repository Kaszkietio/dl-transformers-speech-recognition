import os
import re
from typing import Optional

import librosa
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torchaudio
import torchaudio.transforms as TA
import torchvision.transforms.functional as FV
import torchaudio.functional as FA


WAV_FILE_REGEX = re.compile(".*\.wav")


class AudioToSpectrogramDataset(Dataset):
    def __init__(
            self,
            path: str,
            transform: Optional[nn.Sequential] = None,
            augmentations_path: Optional[str] = None
        ):
        self.root_dir_ = path
        self.classes = os.listdir(path)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        if augmentations_path:
            self.augmentations = True
            self.background_noises = [torch.from_numpy(librosa.load(os.path.join(augmentations_path, f), sr=None)[0])
                                      for f in os.listdir(augmentations_path)
                                      if WAV_FILE_REGEX.match(f)]
        else:
            self.augmentations = False

        if transform:
            self.transform = transform
        else:
            self.transform = nn.Sequential(
                TA.MelSpectrogram(sample_rate=16_000, n_fft=2048, n_mels=128, win_length=256),
                TA.AmplitudeToDB()
            )

        self.audio_files_ = []
        for i, c in enumerate(self.classes):
            files_in_class = [(os.path.join(self.root_dir_, c, f), i)
                              for f in os.listdir(os.path.join(self.root_dir_, c))]
            self.audio_files_.extend(files_in_class)


    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        item_path, class_id = self.audio_files_[index]
        X, sample_rate = librosa.load(item_path, sr=None)
        X = torch.from_numpy(X)
        y = torch.tensor([class_id])

        if self.augmentations and self.classes[class_id] != "silence":
            noise_idx = torch.randint(0, len(self.background_noises), size=(1,)).item()
            noise = self.background_noises[noise_idx]
            noise_offset = torch.randint(0, len(noise) - len(X), size=(1,)).item()
            noise = noise[noise_offset:noise_offset + len(X)]
            X = TA.AddNoise()(X.unsqueeze(0), noise.unsqueeze(0), snr=torch.tensor([10.0])).squeeze(0)

        X = self.transform(X)

        if self.augmentations and self.classes[class_id] != "silence":
            X = TA.FrequencyMasking(freq_mask_param=10)(X)
            X = TA.TimeMasking(time_mask_param=10)(X)

        # Resize to 128x128
        X = FV.resize(X.unsqueeze(0), (128, 128)).squeeze(0)
        assert sample_rate == 16000, f"Sample rate is {sample_rate}, expected 16000"
        return X, y


    def __len__(self):
        return len(self.audio_files_)


if __name__ == "__main__":
    import argparse
    import timeit
    import numpy as np
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--augmentations", required=True)
    config = parser.parse_args()
    print("Path to dataset:", os.path.abspath(config.dataset))
    print("Path to augmentations:", os.path.abspath(config.augmentations))

    seed = 128
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    start = timeit.default_timer()
    dataset = AudioToSpectrogramDataset(config.dataset, augmentations_path=config.augmentations)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)
    X, y = next(iter(dataloader))
    end = timeit.default_timer()

    plt.imshow(X[0].numpy())
    plt.title(f"Class: {dataset.classes[y[0].item()]}")
    plt.show()
    print("Shapes:", X.shape, y.shape)
    print("Mean and std of X:", np.mean(X.numpy()), np.std(X.numpy()))
    print("Dataset size", len(dataset))
    print("Time taken to load dataset:", end - start, "seconds")
    print("Dataset loaded successfully.")