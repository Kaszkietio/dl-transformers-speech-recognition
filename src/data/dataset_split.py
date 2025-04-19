import argparse
import shutil
import os
import numpy as np
from tqdm import tqdm
import librosa
import soundfile as sf


CLASSES = {"yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"}


def process_class_folder(
        audio_folder: str,
        folder_class_name: str,
        class_name: str,
        valid_set: set,
        test_set: set,
        output_path: str
):
    audio_files = os.listdir(os.path.join(audio_folder, folder_class_name))
    for audio_file in tqdm(audio_files):
        if not audio_file.endswith(".wav"):
            print(f"Skipping {audio_file}...")
            continue

        filename = folder_class_name + '/' + audio_file
        if filename in valid_set:
            subset = "valid"
        elif filename in test_set:
            subset = "test"
        else:
            subset = "train"

        src = os.path.join(audio_folder, folder_class_name, audio_file)
        dest = os.path.join(output_path, subset, class_name, folder_class_name + "_" + audio_file)
        shutil.copy(src, dest)


def process_silence_files(
        noise_folder: str,
        output_path: str
):
    class_name = "silence"
    os.makedirs(os.path.join(output_path, "train", class_name), exist_ok=True)
    os.makedirs(os.path.join(output_path, "valid", class_name), exist_ok=True)
    os.makedirs(os.path.join(output_path, "test", class_name), exist_ok=True)

    audio_files = os.listdir(noise_folder)
    for audio_file in tqdm(audio_files):
        if not audio_file.endswith(".wav"):
            print(f"Skipping {audio_file}...")
            continue

        base_filename = audio_file.split('.')[0]
        src = os.path.join(noise_folder, audio_file)

        y, sr = librosa.load(src, sr=None)
        y_split = np.array_split(y, np.arange(sr, len(y), sr))
        if y_split[-1].shape[0] < 16000:
            y_split.pop()

        np.random.shuffle(y_split)
        valid_set_size = len(y_split) // 10
        test_set_size = len(y_split) // 10
        train_set_size = len(y_split) - valid_set_size - test_set_size

        for i, y in enumerate(y_split):
            if i < train_set_size:
                subset = "train"
            elif i < train_set_size + valid_set_size:
                subset = "valid"
            else:
                subset = "test"
            dest = os.path.join(output_path, subset, class_name, base_filename + f"_{i}.wav")
            sf.write(dest, y, sr)


def main(config):

    input_path = config.input
    output_path = config.output

    print("Creating output directory...")
    os.makedirs(output_path, exist_ok=True)
    subsets = ["train", "test", "valid"]
    for subset in subsets:
        os.makedirs(os.path.join(output_path, subset), exist_ok=True)
    print("Creating output directory done.")

    print("Creating train/test/valid directories...")
    os.makedirs(os.path.join(output_path, "train"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "test"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "valid"), exist_ok=True)
    print("Creating train/test/valid directories done.")

    print("Reading train/valid/test list...")
    with open(os.path.join(input_path, "validation_list.txt"), "r") as f:
        valid_set = set([file.strip() for file in f.readlines()])

    with open(os.path.join(input_path, "testing_list.txt"), "r") as f:
        test_set = set([file.strip() for file in f.readlines()])
    print("Reading train/valid/test list done.")


    print("Reading audio files...")
    audio_folder = os.path.join(input_path, "audio")
    folder_class_list = os.listdir(audio_folder)
    for folder_class_name in folder_class_list:

        if folder_class_name == "_background_noise_":
            noise_folder = os.path.join(audio_folder, folder_class_name)
            shutil.copytree(noise_folder, os.path.join(output_path, folder_class_name))
            process_silence_files(noise_folder, output_path)
            continue


        class_name = folder_class_name if folder_class_name in CLASSES else "unknown"
        print(f"Processing class {folder_class_name}/{class_name}...")
        os.makedirs(os.path.join(output_path, "train", class_name), exist_ok=True)
        os.makedirs(os.path.join(output_path, "test", class_name), exist_ok=True)
        os.makedirs(os.path.join(output_path, "valid", class_name), exist_ok=True)

        process_class_folder(audio_folder, folder_class_name, class_name,
                             valid_set, test_set, output_path)
        print(f"Processing class {class_name} done.")

    print("Check if all files are copied...")

    src_files = sum([sum([1 if audio.endswith(".wav") else 0
                          for audio in os.listdir(os.path.join(audio_folder, class_name))])
                          for class_name in os.listdir(audio_folder)])
    dst_files = sum([sum([1 if audio.endswith(".wav") else 0
                          for audio in os.listdir(os.path.join(output_path, subset, class_name))])
                          for subset in subsets for class_name in os.listdir(os.path.join(output_path, subset))])

    print(f"Source files: {src_files}")
    print(f"Destination files: {dst_files}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("dataset-split")
    parser.add_argument("--input", type=str, help="Path to the input dataset", required=True)
    parser.add_argument("--output", type=str, help="Path to the output dataset", required=True)

    config = parser.parse_args()
    main(config)
