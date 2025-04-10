if __name__ != "__main__":
    raise ImportError("This script is not intended to be imported as a module.")

import argparse
import shutil
import os
from tqdm import tqdm

parser = argparse.ArgumentParser("dataset-split")
parser.add_argument("--input", type=str, help="Path to the input dataset")
parser.add_argument("--output", type=str, help="Path to the output dataset")

config = parser.parse_args()

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

print("Test set:", test_set)
print("Validation set:", valid_set)


print("Reading audio files...")
audio_folder = os.path.join(input_path, "audio")
classes_list = os.listdir(audio_folder)
for class_name in classes_list:
    print(f"Processing class {class_name}...")
    os.makedirs(os.path.join(output_path, "train", class_name), exist_ok=True)
    os.makedirs(os.path.join(output_path, "test", class_name), exist_ok=True)
    os.makedirs(os.path.join(output_path, "valid", class_name), exist_ok=True)

    audio_files = os.listdir(os.path.join(audio_folder, class_name))
    for audio_file in tqdm(audio_files):
        if not audio_file.endswith(".wav"):
            print(f"Skipping {audio_file}...")
            continue

        filename = class_name + "/" + audio_file
        if filename in valid_set:
            subset = "valid"
        elif filename in test_set:
            subset = "test"
        else:
            subset = "train"

        src = os.path.join(audio_folder, filename)
        dst = os.path.join(output_path, subset, filename)
        shutil.copy(src, dst)
    print(f"Processing class {class_name} done.")

print("Check if all files are copied...")
src_files = sum([sum([1 if audio.endswith(".wav") else 0 for audio in os.listdir(os.path.join(audio_folder, class_name))])
                    for class_name in os.listdir(audio_folder)])
dst_files = sum([sum([1 if audio.endswith(".wav") else 0 for audio in os.listdir(os.path.join(output_path, subset, class_name))])
                    for subset in subsets for class_name in os.listdir(os.path.join(output_path, subset))])
print(f"Source files: {src_files}")
print(f"Destination files: {dst_files}")