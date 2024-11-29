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
                  test_valid_fractions=[0.05, 0.05],
                  move=False):

    input_ = traverse_dir(input_dir)
    input_labels_ = traverse_dir(input_ / "labels")
    input_images_ = traverse_dir(input_ / "images")

    label_file_list = np.array([f for f in os.listdir(str(input_labels_))])
    test_frac = test_valid_fractions[0]
    valid_frac = test_valid_fractions[1]
    siphoned_frac = test_frac + valid_frac
    leftover_valid_frac = valid_frac / siphoned_frac
    train, test = tt_split(label_file_list, siphoned_frac)
    test, valid = tt_split(test, leftover_valid_frac)
    configs = {
        "test": test,
        "train": train,
        "valid": valid,
    }

    output_ = create_dir(Path(save_dir) / f"{dataset_name}_split")

    for config, label_file_names in tqdm.tqdm(configs.items()):
        output_config_ = create_dir(output_ / config)
        output_config_labels_ = create_dir(output_config_ / "labels")
        output_config_images_ = create_dir(output_config_ / "images")
        for label_file_name in tqdm.tqdm(label_file_names):
            image_file_name = Path(label_file_name).stem + ".jpg"
            shutil.copy(str(input_images_ / image_file_name),
                        output_config_images_)
            shutil.copy(str(input_labels_ / label_file_name),
                        output_config_labels_)

#dataset_name = "DocBank_10000+percentage+shuffle"
#dataset_name = "DocBank_10000+median+shuffle"
#dataset_name = "DocBank_10000+median+sampling"
#dataset_name = "DocBank_10000+percentage+sampling"
dataset_name = "DocBank_10000+percentage+sampling_1000"

split_yolo(
    input_dir=f"C:\\.datasets_converted\\{dataset_name}\\train",
    dataset_name=dataset_name,
    save_dir="C:\\.datasets_converted",
    test_valid_fractions=[0.1, 0.1])
