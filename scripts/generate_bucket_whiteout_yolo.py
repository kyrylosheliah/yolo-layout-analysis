import shutil
from pathlib import Path
from PIL import Image, ImageDraw

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

def dataset_whiteout_normalization(dataset_name, path_prefix="", restore_images=False):

    counters = [
        0, # 0
        0, # 1
        0, # 2
        0, # 3
        0, # 4
        0, # 5
        0, # 6
        0, # 7
        0, # 8
        0, # 9
        0, # 10
        0, # 11
        0, # 12
    ]
    counters_max = 0
    def is_uncapped_class(class_index):
        nonlocal counters_max
        nonlocal counters
        could_become = 1 + counters[class_index]
        if could_become > counters_max:
            for counter in counters:
                if counter < counters_max:
                    return False
            counters_max = could_become
        counters[class_index] = could_become
        return True
    def whiteout_image_patch(draw, width, height, cx, cy, w, h):
        box_width = width * w
        box_height = height * h
        x_center = width * cx
        y_center = height * cy
        x0 = x_center - (box_width / 2.0)
        x1 = x0 + box_width
        x1 = x_center + (box_width / 2.0)
        y0 = y_center - (box_height / 2.0)
        y1 = y0 + box_height
        y1 = y_center + (box_height / 2.0)
        shape = (int(x0), int(y0), int(x1), int(y1))
        draw.rectangle(shape, fill="white")
        #draw.rectangle(shape, outline="red")
        #draw.line(shape, fill=(255, 255, 0), width=10)

    dataset_ = traverse_dir(traverse_dir(f"{path_prefix}datasets/") / dataset_name)
    output_ = create_dir(create_dir(f"{path_prefix}datasets_converted/") / dataset_name)
    print(f"dataset '{dataset_}' | output: '{output_}'")
    shutil.copy(dataset_ / "data.yaml", output_)

    for task_path in sorted(Path(dataset_).resolve().glob("*")):
        if not task_path.is_dir():
            continue
        task = task_path.stem
        dataset_task_ = traverse_dir(dataset_ / task)
        dataset_task_labels_ = traverse_dir(dataset_task_ / "labels")
        dataset_task_images_ = traverse_dir(dataset_task_ / "images")
        output_task_ = create_dir(output_ / task)
        output_task_labels_ = create_dir(output_task_ / "labels")
        output_task_images_ = create_dir(output_task_ / "images")
        print(f"\ntask: '{task}'")
        visual_i = 0
        for input_labels_file in sorted(Path(dataset_task_labels_).resolve().glob("*.txt")):
            # prepare result paths
            stem = input_labels_file.stem
            print(f"{visual_i} ", end="")
            visual_i += 1
            image_name = stem + ".jpg"
            output_labels_path = output_task_labels_ / (stem + ".txt")
            input_image_path = dataset_task_images_ / image_name
            output_image_path = output_task_images_ / image_name
            # prepare an image
            if restore_images or not Path(output_image_path).exists():
                shutil.copy(input_image_path, output_task_images_)
            try:
                output_image = Image.open(output_image_path)
            except:
                continue
            width, height = output_image.size
            draw = ImageDraw.Draw(output_image)
            need_to_write_image = False
            # process annotations
            with open(input_labels_file, "r") as file:
                rows = file.read().split("\n")
            for irow in range(len(rows)):
                row = rows[irow]
                if len(row) == 0:
                    continue
                components = row.split(" ")
                obj_class = int(components[0])
                if is_uncapped_class(obj_class):
                    continue
                else:
                    need_to_write_image = True
                rows[irow] = ""
                whiteout_image_patch(draw, float(width), float(height),
                                     cx = float(components[1]),
                                     cy = float(components[2]),
                                     w = float(components[3]),
                                     h = float(components[4]))
            # write processed annotation
            with open(output_labels_path, "w") as file:
                rows_filtered = [r for r in rows if r != ""]
                file.write("\n".join(rows_filtered))
            if need_to_write_image:
                output_image.save(output_image_path)
#TESTDATASET
#DocBank
dataset_whiteout_normalization(dataset_name="DocBank", path_prefix="C:/.", restore_images=True)
