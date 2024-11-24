from pathlib import Path
from PIL import Image, ImageDraw
import tqdm
import json

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

def generate_dataset_from_layouts(input_dir, dataset_name, output_dir, page_width=640, page_height=640):
    W, H = page_width, page_height
    input_ = traverse_dir(traverse_dir(input_dir) / dataset_name)
    input_layouts_ = traverse_dir(input_ / "layouts")
    output_ = create_dir(traverse_dir(output_dir) / dataset_name)
    output_train_ = create_dir(output_ / "train")
    output_train_labels_ = create_dir(output_train_ / "labels")
    output_train_images_ = create_dir(output_train_ / "images")
    for layout_file_name in tqdm.tqdm(sorted(Path(input_layouts_).resolve().glob("*.json"))):
        stem = layout_file_name.stem
        #image = Image.new("RGBA", (W, H), (255,255,255,255))
        image = Image.new("RGB", (W, H), (255,255,255))
        labels = []
        with open(layout_file_name, "r") as file:
            patch_list = json.loads(file.read())
        for patch in patch_list:
            cx, cy, w, h = patch["cx"], patch["cy"], patch["w"], patch["h"]
            class_id = patch["class_id"]
            labels.append(f"{class_id} {cx} {cy} {w} {h}")
            patch = Image.open(patch["file_path"]).resize((int(W * w), int(H * h)))
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
        output_path = str(output_train_images_ / f"{stem}.jpg")
        image.save(output_path)
        labels = "\n".join(labels)
        with open(str(output_train_labels_ / f"{stem}.txt"), "w") as file:
            file.write(labels)

generate_dataset_from_layouts(
    input_dir="C:\\.layouts\\",
    dataset_name="DocBank_10000_layouts+percentage+shuffle",
    output_dir="C:\\.datasets_converted\\")
