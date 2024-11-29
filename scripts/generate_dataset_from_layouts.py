from pathlib import Path
from PIL import Image, ImageDraw
import cv2
import tqdm
import json
import albumentations as A
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

def per_patch_transform(w, h):
    return A.Compose([
        A.RandomResizedCrop(
            height=h, width=w,
            scale=(0.9,0.9),
            ratio=(w/h,w/h),
            p=0.25),
        A.ElasticTransform(alpha=1, sigma=10),
        A.GridDistortion(num_steps=5, distort_limit=(-0.2,0.2)),
    ])
def per_page_transform(W, H):
    return A.Compose([
        A.RandomBrightnessContrast(
            brightness_limit=(-0.1, 0.1),
            contrast_limit=(-0.1, 0.1),
            p=1.0),
        A.ColorJitter(
            brightness=0, contrast=0,
            saturation=(0.5, 1.5), hue=(-0.25, 0.25)),
        A.GaussNoise(var_limit=(0,400)),
        A.OneOf([
            A.MedianBlur(blur_limit=3, p=1.0),
            A.Blur(blur_limit=3, p=1.0),
        ]),
    ])

def transform_pil(image, transform):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    image = transform(image=image)["image"]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    return image

def generate_dataset_from_layouts(input_dir, dataset_name, output_dir, page_size=640):
    W, H = page_size, page_size
    input_ = traverse_dir(traverse_dir(input_dir) / dataset_name)
    input_layouts_ = traverse_dir(input_ / "layouts")
    size_suffix = "" if page_size == 640 else f"_{page_size}"
    output_ = create_dir(traverse_dir(output_dir) / f"{dataset_name}{size_suffix}")
    output_train_ = create_dir(output_ / "train")
    output_train_labels_ = create_dir(output_train_ / "labels")
    output_train_images_ = create_dir(output_train_ / "images")
    page_transform = per_page_transform(W, H)
    for layout_file_name in tqdm.tqdm(sorted(Path(input_layouts_).resolve().glob("*.json"))):
        stem = layout_file_name.stem
        #image = Image.new("RGBA", (W, H), (255,255,255,255))
        image = Image.new("RGB", (W, H), (255,255,255))
        labels = []
        with open(layout_file_name, "r") as file:
            patch_list = json.loads(file.read())
        for patch in patch_list:
            cx, cy, w, h = patch["cx"], patch["cy"], patch["w"], patch["h"]
            p_W, p_H = int(W * w), int(H * h)
            class_id = patch["class_id"]
            labels.append(f"{class_id} {cx} {cy} {w} {h}")
            patch = Image.open(patch["file_path"]).resize((p_W, p_H))
            patch_transform = per_patch_transform(p_W, p_H)
            patch = transform_pil(patch, patch_transform)
            patch_x1 = int(W * (cx - (w / 2)))
            patch_x2 = int(H * (cy - (h / 2)))
            image.paste(patch, (patch_x1, patch_x2))
            #if True: ### debug
            #    overlay = Image.new("RGBA", image.size, (0,0,0,0))
            #    draw = ImageDraw.Draw(overlay)
            #    x1 = int(W * (cx - (w / 2)))
            #    x2 = int(W * (cx + (w / 2)))
            #    y1 = int(H * (cy - (h / 2)))
            #    y2 = int(H * (cy + (h / 2)))
            #    fill = (
            #        255,#255 if i % 3 == 0 else 0,
            #        0,#255 if i % 3 == 1 else 0,
            #        0,#255 if i % 3 == 2 else 0,
            #        63
            #    )
            #    draw.rectangle([(x1, y1), (x2, y2)], fill=fill)
            #    image = Image.alpha_composite(image, overlay)
        image = transform_pil(image, page_transform)
        output_path = str(output_train_images_ / f"{stem}.jpg")
        image.save(output_path)
        labels = "\n".join(labels)
        with open(str(output_train_labels_ / f"{stem}.txt"), "w") as file:
            file.write(labels)

generate_dataset_from_layouts(
    input_dir="C:\\.layouts\\",
    #dataset_name="DocBank_10000+percentage+shuffle",
    #dataset_name="DocBank_10000+median+shuffle",
    #dataset_name="DocBank_10000+median+sampling",
    dataset_name="DocBank_10000+percentage+sampling",
    output_dir="C:\\.datasets_converted\\",
    #page_size=640)
    page_size=1000)
