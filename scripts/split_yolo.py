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

def tt_split(x, shard_size=0.1):
    i = int((1 - shard_size) * x.shape[0])
    o = np.random.permutation(x.shape[0])
    x_left, x_right = np.split(np.take(x, o, axis=0), [i])
    return x_left, x_right

def split_yolo(input_dir="",
                  dataset_name="",
                  save_dir="",
                  save_dataset_name="",
                  test_valid_fractions=[0.05, 0.05],
                  move=False):

    dataset_ = traverse_dir(Path(input_dir) / dataset_name)
    dataset_labels_ = traverse_dir(dataset_ / "labels")
    dataset_images_ = traverse_dir(dataset_ / "images")

    label_file_list = np.array([f for f in os.listdir(str(dataset_labels_))])
    test_frac = test_valid_fractions[0]
    valid_frac = test_valid_fractions[1]
    train, test = tt_split(label_file_list, test_frac + valid_frac)
    test, valid = tt_split(test, valid_frac)
    configs = {
        "test": test,
        "train": train,
        "valid": valid,
    }

    output_ = create_dir(Path(save_dir) / save_dataset_name)

    for config, label_file_names in tqdm.tqdm(configs.items()):
        output_config_ = create_dir(output_ / config)
        output_config_labels_ = create_dir(output_config_ / "labels")
        output_config_images_ = create_dir(output_config_ / "images")
        if move:
            for label_file_name in tqdm.tqdm(label_file_names):
                image_file_name = Path(label_file_name).stem + ".jpg"
                shutil.move(str(dataset_images_ / image_file_name),
                            output_config_images_)
                shutil.move(str(dataset_labels_ / label_file_name),
                            output_config_labels_)
        else:
            for label_file_name in tqdm.tqdm(label_file_names):
                image_file_name = Path(label_file_name).stem + ".jpg"
                shutil.copy(str(dataset_images_ / image_file_name),
                            output_config_images_)
                shutil.copy(str(dataset_labels_ / label_file_name),
                            output_config_labels_)

split_yolo(
    input_dir="C:\\.datasets_converted",
    dataset_name="DocSynth300K",
    save_dir="C:\\.datasets_converted",
    save_dataset_name="DocSynth300K_split",
    test_valid_fractions=[0.05, 0.05],
    move=True)
