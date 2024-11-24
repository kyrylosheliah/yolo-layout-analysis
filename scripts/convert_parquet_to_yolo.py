import io
import os
import tqdm
import pandas as pd
from PIL import Image
from pathlib import Path, PurePath

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

def convert_docsynth_parquet(input_dir="", save_dir=""):
    dataset_name = "DocSynth300K"
    dataset_ = create_dir(Path(save_dir) / dataset_name)

    parquet_dir = traverse_dir(Path(input_dir) / dataset_name)
    parquet_list = os.listdir(parquet_dir)
    parquet_list = [f for f in parquet_list if "parquet" in f]

    dataset_labels_ = create_dir(dataset_ / "labels")
    dataset_images_ = create_dir(dataset_ / "images")
    train_txt = open(str(dataset_ / "train300k.txt"), "w")

    for parquet in tqdm.tqdm(parquet_list):
        df = pd.read_parquet(str(parquet_dir / parquet))
        for _, row in tqdm.tqdm(df.iterrows()):
            filename = row['filename']
            image_data = row['image_data']
            anno_string = row['anno_string']
            image = Image.open(io.BytesIO(image_data))
            # save image / anno / txt
            train_txt.write(str(dataset_images_ / filename) + "\n")
            image.save(str(dataset_images_ / filename))
            stem = PurePath(str(filename)).stem
            with open(str(dataset_labels_ / (stem + ".txt")), "w") as f:
                f.write("\n".join(anno_string))

    # write validation file (dummy)
    train_txt.close()
    image_list = list(open(str(dataset_ / "train300k.txt"), "r").readlines())[:1000]
    val = open(str(dataset_ / "val.txt"), "w")
    for line in image_list:
        val.write(line)
    val.close()

convert_docsynth_parquet(
    "D:\\.dev\\yolo-layout-analysis\\dependencies",
    "C:\\.datasets_converted")
