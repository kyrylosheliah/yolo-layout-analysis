import os
import tqdm
from pathlib import Path
import shutil
import numpy as np

def traverse_dir(dir):
    dir = Path(f"{dir}")
    if not dir.exists():
        raise Exception(f"Path not found: {str(dir)}")
    return dir

def create_dir(dir):
    dir = Path(f"{dir}")
    if not dir.exists():
        dir.mkdir()
    return dir

def print_class_count(input_dir="", dataset_name=""):
    dataset_ = traverse_dir(Path(input_dir) / dataset_name)
    max = 0
    for config in tqdm.tqdm(os.listdir(str(dataset_))):
        dataset_config_ = traverse_dir(dataset_ / config)
        if not dataset_config_.is_dir():
            continue
        dataset_config_labels_ = traverse_dir(dataset_config_ / "labels")
        for label_file in tqdm.tqdm(os.listdir(str(dataset_config_labels_))):
            label_file_path = str(dataset_config_labels_ / label_file)
            with open(label_file_path, "r") as file:
                rows = [row for row in file.read().split("\n") if len(row) > 0]
            for row in rows:
                class_idx = int(row.split(" ")[0])
                if class_idx > max:
                    max = class_idx
    return max

class_count = print_class_count(input_dir="C:\\.datasets",
                                dataset_name="DocSynth300K_split")

print(f"maximum class index is: {class_count}, or {class_count + 1} classes encountered")