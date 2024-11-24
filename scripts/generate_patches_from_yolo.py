import json
import random
from PIL import Image
from pathlib import Path
import tqdm

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

def cut_out_image_patch(image, W, H, cx, cy, w, h):
    box_width = W * w
    box_height = H * h
    x_center = W * cx
    y_center = H * cy
    x0 = x_center - (box_width / 2.0)
    x1 = x0 + box_width
    x1 = x_center + (box_width / 2.0)
    y0 = y_center - (box_height / 2.0)
    y1 = y0 + box_height
    y1 = y_center + (box_height / 2.0)
    shape = (int(x0), int(y0), int(x1), int(y1))
    return image.crop(shape)
    #draw.rectangle(shape, outline="red")
    #draw.line(shape, fill=(255, 255, 0), width=10)

class patch_metadata:
    def __init__(self, cx, cy, w, h):
        self.cx = cx
        self.cy = cy
        self.w = w
        self.h = h

def generate_patches_from_yolo(input_dir, dataset_name, config, patch_count, output_dir):
    input_ = traverse_dir(traverse_dir(input_dir) / dataset_name)
    # get class definitions from dataset yaml file
    class_definitions = extract_class_definitions(str(input_))
    output_ = create_dir(traverse_dir(output_dir) / f"{dataset_name}_{patch_count}_patches")
    # then ensure class directories exist
    for i in range(len(class_definitions)):
        create_dir(output_ / f"{i}")
    # then write a definition file
    with open(str(output_ / "class_definitions.txt"), "w") as file:
        file.write("\n".join(class_definitions))
    input_config_ = traverse_dir(input_ / config)
    input_config_labels_ = traverse_dir(input_config_ / "labels")
    input_config_images_ = traverse_dir(input_config_ / "images")
    i = 0
    selected = list(Path(input_config_labels_).resolve().glob("*.txt"))
    random.shuffle(selected)
    for image_labels_path in tqdm.tqdm(selected):
        stem = image_labels_path.stem
        image_path = input_config_images_ / f"{stem}.jpg"
        image = Image.open(image_path)
        H, W = image.height, image.width
        with open(image_labels_path, "r") as file:
            rows = file.read().split("\n")
        for row in rows:
            if len(row) == 0:
                continue
            components = row.split(" ")
            cx = float(components[1])
            cy = float(components[2])
            w = float(components[3])
            h = float(components[4])
            if w < 0.01 or h < 0.01:
                continue
            patch_class = components[0]
            image_patch = cut_out_image_patch(image, W, H, cx, cy, w, h)
            image_patch_filename = output_ / patch_class / f"{i}.jpg"
            image_patch.save(str(image_patch_filename))
            image_metadata_filename = output_ / patch_class / f"{i}.json"
            image_metadata = patch_metadata(cx, cy, w, h)
            with open(str(image_metadata_filename), "w") as file:
                file.write(json.dumps(image_metadata, default=lambda k: k.__dict__))
            i += 1
            if i >= patch_count:
                return

generate_patches_from_yolo(
    input_dir="C:\\.datasets\\",
    dataset_name="DocBank",
    config="valid",
    patch_count=40000,
    output_dir="C:\\.patches\\",
)