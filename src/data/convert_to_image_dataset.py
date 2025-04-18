import os
import PIL.Image
from PIL.Image import Image
from torch.utils.data import DataLoader
import torchvision.transforms.functional as FV
import matplotlib.pyplot as plt

from spectrogram_dataset import AudioToSpectrogramDataset

def convert_audio_to_image(audio_path: str, output_path: str, augmentations_path: str = None, repetition_num: int = 1):
    """
    Convert audio files to image files using the AudioToSpectrogramDataset class.
    Args:
        audio_path (str): Path to the audio dataset.
        output_path (str): Path to save the converted image dataset.
        augmentations_path (str, optional): Path to the augmentations dataset.
    """
    subsets = ["train", "test", "valid"]
    os.makedirs(args.output_path, exist_ok=True)


    for subset in subsets:
        print("Creating output directory for subset:", subset)
        os.makedirs(os.path.join(output_path, subset), exist_ok=True)
        audio_subset_path = os.path.join(audio_path, subset)
        output_subset_path = os.path.join(output_path, subset)

        dataset = AudioToSpectrogramDataset(audio_subset_path, augmentations_path=augmentations_path)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
        print("Creating output directory for classes:", dataset.classes)
        for class_name in dataset.classes:
            os.makedirs(os.path.join(output_subset_path, class_name), exist_ok=True)


        for repetition in range(repetition_num):
            print("------------Repetition:", repetition + 1, "------------")
            for i, (audio, label) in enumerate(loader):
                print(f"[{i}] Audio shape:", audio.shape, "Label:", label.item())
                # Convert the audio tensor to a PIL image and save it
                # image: Image = PIL.Image.fromarray(audio.numpy())
                # plt.imshow(image, cmap='gray')
                # plt.show()
                # image = image.convert("RGB")
                # plt.imshow(image, cmap='gray')
                # plt.show()
                path = os.path.join(output_subset_path, dataset.classes[int(label.item())],
                                    f"{repetition}_audio_{i}.png")
                plt.imsave(path, audio.squeeze(0))

                # image.save(os.path.join(output_subset_path, dataset.classes[int(label.item())],
                #                         f"{repetition}_audio_{i}.png"))


if __name__ == "__main__":
    import os
    import argparse
    import random
    import numpy as np
    import torch

    parser = argparse.ArgumentParser(description="Convert a dataset to an image dataset")
    parser.add_argument("--audio_dataset_path", type=str, required=True, help="Path to the audio dataset")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output image dataset")
    parser.add_argument("--augmentations_path", type=str, help="Path to the augmentations dataset")
    parser.add_argument("--repetition_num", type=int, default=1, help="Number of repetitions for each audio file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()
    seed = args.seed
    random.seed(seed)
    np.random.RandomState(seed)
    torch.manual_seed(seed)

    convert_audio_to_image(args.audio_dataset_path, args.output_path, args.augmentations_path, args.repetition_num)
    print(f"Converted audio dataset from {args.audio_dataset_path} to image dataset at {args.output_path}")


