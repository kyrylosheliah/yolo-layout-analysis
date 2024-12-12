import random
from pathlib import Path
import shutil
import tqdm

def traverse_dir(dir):
    dir = Path(f"{dir}")
    if not dir.exists():
        raise Exception(f"Path not found: {str(dir)}")
    return dir

def file_has_content(file_name):
    with open(file_name, "r") as file:
        contents = file.read()
    return len(contents) > 10

def siphon(from_dir, count, to_dir, check_labels_emptiness=True):

    from_ = traverse_dir(from_dir)
    from_labels_ = traverse_dir(from_ / "labels")
    from_images_ = traverse_dir(from_ / "images")
    to_ = traverse_dir(to_dir)
    to_labels = traverse_dir(to_ / "labels")
    to_images = traverse_dir(to_ / "images")

    selected = sorted(Path(from_labels_).resolve().glob("*.txt"))
    if check_labels_emptiness:
        selected = [fname for fname in selected if file_has_content(fname)]
    if len(selected) < count:
        raise Exception(f"Error: can't select {count} images when there are only {len(selected)} of them")
    selected = random.sample(selected, count)
    for input_labels_filename in tqdm.tqdm(selected):
        stem = input_labels_filename.stem
        output_labels_filename = to_labels / f"{stem}.txt"
        if not output_labels_filename.exists():
            output_labels_filename = str(output_labels_filename)
            input_labels_filename = str(from_labels_ / f"{stem}.txt")
            shutil.move(input_labels_filename, output_labels_filename)
    for input_labels_filename in tqdm.tqdm(selected):
        stem = input_labels_filename.stem
        output_image_filename = to_images / f"{stem}.jpg"
        if not output_image_filename.exists():
            output_image_filename = str(output_image_filename)
            input_image_filename = str(from_images_ / f"{stem}.jpg")
            shutil.move(input_image_filename, output_image_filename)

siphon(
    from_dir="C:\\.datasets\\DocBank\\train",
    count=10000,
    to_dir="C:\\.datasets\\DocBank\\valid",
)
