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

def extract_class_definitions(dataset_dir):
    with open(str(Path(dataset_dir) / "data.yaml"), "r") as file:
        rows = file.read().split("\n")
    class_count = None
    class_names = None
    for row in rows:
        if row.startswith("nc:"):
            class_count = int(row.removeprefix("nc:"))
        elif row.startswith("names:"):
            class_names = row.removeprefix("names:")
    if class_names != None:
        class_names = class_names.strip(" ").lstrip("[").rstrip("]")
        class_names = class_names.split(",")
        class_names = [name.strip(" ").strip('"').strip("'").strip('"')
                       for name in class_names]
        print(class_names)
        return class_names
    if class_count != None:
        return [i for i in range(class_count)]
    print("'class_count:<NumberOfClasses>' row not found in the data.yaml file")
    return None

def rearrange_classes(input_dir="", dataset_name="", output_dir="", target_arrangement=[]):
    input_ = Path(input_dir) / dataset_name
    output_ = create_dir(traverse_dir(output_dir) / dataset_name)
    class_definitions = extract_class_definitions(input_)
    class_remap = []
    for _, class_name in enumerate(class_definitions):
        class_remap.append(str(target_arrangement.index(class_name)))
    for config in tqdm.tqdm(os.listdir(str(input_))):
        dataset_config_ = traverse_dir(input_ / config)
        if not dataset_config_.is_dir():
            continue
        output_config_ = create_dir(output_ / config)
        dataset_config_labels_ = traverse_dir(dataset_config_ / "labels")
        output_config_labels_ = create_dir(output_config_ / "labels")
        for label_file in tqdm.tqdm(dataset_config_labels_.resolve().glob("*.txt")):
            label_file_path = str(dataset_config_labels_ / label_file)
            with open(label_file_path, "r") as file:
                rows = [row for row in file.read().split("\n") if len(row) > 0]
            for j, row in enumerate(rows):
                row_split = row.split(" ")
                class_idx = int(row_split[0])
                row_split[0] = class_remap[class_idx]
                rows[j] = " ".join(row_split)
            stem = label_file.stem
            with open(str(output_config_labels_ / f"{stem}.txt"), "w") as file:
                file.write("\n".join(rows))

target_arrangement = [
    "title",
    "plain text",
    "abandon",
    "figure",
    "figure_caption",
    "table",
    "table_caption",
    "table_footnote",
    "isolate_formula",
    "formula_caption",
]

class_count = rearrange_classes(input_dir="C:\\.datasets",
                                dataset_name="custom",
                                output_dir="C:\\.datasets_converted",
                                target_arrangement=target_arrangement)
