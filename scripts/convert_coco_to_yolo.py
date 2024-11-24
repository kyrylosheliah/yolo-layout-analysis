import json
from collections import defaultdict
import shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm

def create_dir(dir):
    dir = Path(f"{dir}")
    if not dir.exists():
        dir.mkdir()
    return dir

def convert_coco_json(dataset_name, json_config_dir, image_dir, save_dir="datasets_converted"):
    """Converts COCO JSON format to YOLO label format"""

    save_dir = create_dir(save_dir)
    dataset_ = create_dir(save_dir / dataset_name)

    # Import json
    for json_file in sorted(Path(json_config_dir).resolve().glob("*.json")):
        json_config_name = json_file.stem
        dataset_config_ = create_dir(dataset_ / json_config_name)
        dataset_config_labels_ = create_dir(dataset_config_ / "labels")
        dataset_config_images_ = create_dir(dataset_config_ / "images")
        with open(json_file) as f:
            data = json.load(f)

        # Create image dict
        images = {"{:g}".format(x["id"]): x for x in data["images"]}
        # Create image-annotations dict
        imgToAnns = defaultdict(list)
        for ann in data["annotations"]:
            imgToAnns[ann["image_id"]].append(ann)

        # Write labels file
        for img_id, anns in tqdm(imgToAnns.items(), desc=f"Annotations {json_config_name}"):
            img = images[f"{img_id:g}"]
            h, w, file_name = img["height"], img["width"], img["file_name"]

            bboxes = []
            for ann in anns:
                if ann["iscrowd"]:
                    continue
                # The COCO box format is [top left x, top left y, width, height]
                box = np.array(ann["bbox"], dtype=np.float64)
                box[:2] += box[2:] / 2  # xy top-left corner to center
                box[[0, 2]] /= w  # normalize x
                box[[1, 3]] /= h  # normalize y
                if box[2] <= 0 or box[3] <= 0:  # if w <= 0 and h <= 0
                    continue

                cls = ann["category_id"] - 1  # class
                box = [cls] + box.tolist()
                if box not in bboxes:
                    bboxes.append(box)

            # Write labels
            with open((dataset_config_labels_ / file_name).with_suffix(".txt"), "w") as file:
                for i in range(len(bboxes)):
                    line = (*(bboxes[i]),)  # cls, box or segments
                    file.write(("%g " * len(line)).rstrip() % line + "\n")

            # Copy an associated image
            if not Path(dataset_config_images_ / file_name).exists():
                shutil.copy(f"{image_dir}/{file_name}",
                            dataset_config_images_)

convert_coco_json(
    dataset_name="DocBank",
    json_config_dir="D:\\.dev\\yolo-layout-analysis-data\\datasets_repos\\DocBank\\MSCOCO_Format_Annotation_clean\\",
    image_dir="D:\\.dev\\yolo-layout-analysis-data\\datasets_repos\\DocBank\\500K_ori_img\\",
    save_dir="C:\\.datasets_converted\\")
