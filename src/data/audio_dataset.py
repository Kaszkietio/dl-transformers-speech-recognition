import re
import os
from typing import Optional
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torchaudio
import librosa

wav_file_regex = re.compile(".*\.wav")

class AudioDataset(Dataset):
    def __init__(
            self,
            path: str,
            transform:Optional[nn.Sequential] = None,
            target_transform:Optional[nn.Sequential] = None
        ):
        self.root_dir_ = path
        self.classes_ = os.listdir(path)
        self.transform = transform
        self.target_transform = target_transform

        self.audio_files_ = []
        for i, c in enumerate(self.classes_):
            files_in_class = [(os.path.join(self.root_dir_, c, f), i)
                              for f in os.listdir(os.path.join(self.root_dir_, c))]
            self.audio_files_.extend(files_in_class)


    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        item_path, class_id = self.audio_files_[index]
        X, sample_rate = librosa.load(item_path, sr=None)
        X = torch.from_numpy(X)
        y = torch.tensor([class_id])
        if self.transform:
            X = self.transform(X)
        if self.target_transform:
            y = self.target_transform(y)

        assert sample_rate == 16000, f"Sample rate is {sample_rate}, expected 16000"
        return torch.from_numpy(X), torch.tensor([class_id])


    def __len__(self):
        return len(self.audio_files_)


def get_train_dataloader(
        path: str,
        batch_size: int = 256,
        transform: Optional[nn.Sequential] = None,
        target_transform: Optional[nn.Sequential] = None
    ) -> DataLoader:
    train_path = os.path.join(path, "train")
    dataset = AudioDataset(train_path, transform=transform, target_transform=target_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    return dataloader


def get_augmentations(augmentations: dict[str]) -> nn.Sequential:
    transforamations = []
    if "time_shift" in augmentations:
        transforamations.append(torch)

    return nn.Sequential(*transforamations)



if __name__ == "__main__":
    import argparse
    import timeit
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument("--path")
    config = parser.parse_args()
    print("Path to dataset:", os.path.abspath(config.path))

    start = timeit.default_timer()
    dataset = AudioDataset(config.path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)
    X, y = next(iter(dataloader))

    end = timeit.default_timer()
    print("Shapes:", X.shape, y.shape)
    print("Mean and std of X:", np.mean(X.numpy()), np.std(X.numpy()))
    print("Dataset size", len(dataset))
    print("Time taken to load dataset:", end - start, "seconds")
    print("Dataset loaded successfully.")