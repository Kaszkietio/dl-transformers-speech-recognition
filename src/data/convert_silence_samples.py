import os
import librosa
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
from torch import nn
import torchaudio.transforms as TA
from torchvision.transforms import functional as FV
from tqdm import tqdm


def convert_audio_to_image(
        audio_path: str,
        output_path: str,
        frames_per_sec: int,
        repetition_num: int
):
    """
    Convert audio files to image files using librosa and matplotlib.
    Args:
        audio_path (str): Path to the audio dataset.
        output_path (str): Path to save the converted image dataset.
        repetition_num (int): Number of times to repeat the conversion for each audio file.
    """
    os.makedirs(output_path, exist_ok=True)

    transform = nn.Sequential(
        TA.Vol(0.5),
        TA.MelSpectrogram(sample_rate=16_000, n_fft=2048, n_mels=128, win_length=256),
        TA.AmplitudeToDB(),
        TA.FrequencyMasking(freq_mask_param=10),
        TA.TimeMasking(time_mask_param=10)
    )

    frame_diff = 16_000 // frames_per_sec

    # Get all audio files in the directory
    audio_files = [f for f in os.listdir(audio_path) if f.endswith('.wav')]
    print("Audio files found:", len(audio_files))
    print("Audio files:", audio_files)\

    for repetition in range(repetition_num):
        print("------------Repetition:", repetition + 1, "------------")
        for i, audio_file in enumerate(tqdm(audio_files)):
            # Load the audio file
            y, sr = librosa.load(os.path.join(audio_path, audio_file), sr=16_000)
            l = len(y)
            y = torch.from_numpy(y)

            # Convert to spectrogram
            for j in range(0, l, frame_diff):
                if j + sr >= l:
                    break
                cur_frame = y[j:j + sr]

                spectrogram = transform(cur_frame)
                spectrogram = FV.resize(spectrogram.unsqueeze(0), (128, 128)).squeeze(0).numpy()

                # Save the spectrogram as an image
                # plt.axis('off')
                output_file = os.path.join(output_path, f"{repetition}_silence_{i}_{j}.png")
                plt.imsave(output_file, spectrogram)


if __name__ == "__main__":
    import argparse

    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_path", required=True, help="Path to the audio dataset")
    parser.add_argument("--output_path", required=True, help="Path to save the converted image dataset")
    parser.add_argument("--frames_per_sec", type=int, default=10, help="Frames per second for the spectrogram")
    parser.add_argument("--repetition_num", type=int, default=1, help="Number of repetitions for each audio file")
    config = parser.parse_args()

    convert_audio_to_image(
        audio_path=config.audio_path,
        output_path=config.output_path,
        frames_per_sec=config.frames_per_sec,
        repetition_num=config.repetition_num
    )
    print("Conversion completed successfully.")