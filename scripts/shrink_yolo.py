import random
from pathlib import Path
import shutil
from pathlib import Path
from PIL import Image, ImageDraw
import tqdm
import shutil

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

def file_has_content(file_name):
    with open(file_name, "r") as file:
        contents = file.read()
    return len(contents) > 10

def shrink(input_dir, dataset_name, count_map, output_dir, image_restoration_dir=None, check_labels_emptiness=False):

    input_ = traverse_dir(Path(input_dir) / dataset_name)
    output_ = create_dir(traverse_dir(output_dir) / dataset_name)
    shutil.copy(input_ / "data.yaml", output_)

    for task_path in tqdm.tqdm(sorted(Path(input_).resolve().glob("*"))):
        if not task_path.is_dir():
            continue
        config = task_path.stem
        count = None
        try:
            count = count_map[config]
        except:
            continue
        input_config_ = traverse_dir(input_ / config)
        input_config_labels_ = traverse_dir(input_config_ / "labels")
        input_config_images_ = traverse_dir(input_config_ / "images")
        output_config_ = create_dir(output_ / config)
        output_config_labels_ = create_dir(output_config_ / "labels")
        output_config_images_ = create_dir(output_config_ / "images")
        selected = sorted(Path(input_config_labels_).resolve().glob("*.txt"))
        if check_labels_emptiness:
            selected = [fname for fname in selected if file_has_content(fname)]
        written_count = len(list(Path(output_config_labels_).resolve().glob("*.txt")))
        if count <= written_count:
            continue
        if len(selected) > count:
            selected = random.sample(selected, count)
        for input_labels_filename in tqdm.tqdm(selected):
            stem = input_labels_filename.stem
            output_labels_filename = output_config_labels_ / f"{stem}.txt"
            if not output_labels_filename.exists():
                output_labels_filename = str(output_labels_filename)
                input_labels_filename = str(input_config_labels_ / f"{stem}.txt")
                shutil.copy(input_labels_filename, output_labels_filename)
        for input_labels_filename in tqdm.tqdm(selected):
            stem = input_labels_filename.stem
            input_image_filename = input_config_images_ / f"{stem}.jpg"
            if input_image_filename.exists():
                input_image_filename = str(input_image_filename)
            else:
                input_image_filename = Path(image_restoration_dir) / f"{stem}.jpg"
            output_image_filename = output_config_images_ / f"{stem}.jpg"
            if not output_image_filename.exists():
                output_image_filename = str(output_image_filename)
                shutil.copy(input_image_filename, output_image_filename)

count_map = {
    "train": 50000,
    "valid": 10000,
    "test": 10000
}

#shrink(
#    input_dir="C:\\.datasets\\",
#    dataset_name="DocBank",
#    count_map=count_map,
#    output_dir="C:\\.datasets_converted\\",
#    image_restoration_dir="D:\\.dev\\yolo-layout-analysis-repos\\DocBank\\500K_ori_img\\",
#    check_labels_emptiness=True,
#)

#shrink(
#    input_dir="C:\\.datasets\\",
#    dataset_name="DocSynth300K_split",
#    count_map=count_map,
#    output_dir="C:\\.datasets_converted\\"
#)

shrink(
    input_dir="D:\\.dev\\yolo-layout-anaysis-datasets\\",
    dataset_name="DocBank_bucket_whiteout",
    count_map=count_map,
    output_dir="C:\\.datasets_converted\\",
    check_labels_emptiness=True,
)