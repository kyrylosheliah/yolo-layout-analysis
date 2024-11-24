import multiprocessing
import random
import json
import itertools
import shutil
import torch
import torchvision
from pathlib import Path
import tqdm

if torch.cuda.is_available():
    device = torch.device('cuda')
device = torch.device("cuda:0")

class element(object):
    def __init__(self, cx, cy, h, w, class_id, file_path):
        self.cx = cx
        self.cy = cy
        self.h = h
        self.w = w
        self.class_id = class_id
        self.file_path = file_path


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

def either_dimension_is_small(element):
    global median_strategy
    if median_strategy:
        return element.w < median_w or element.h < median_h
    else:
        return element.w < 0.05 or element.h < 0.05
def both_dimensions_are_small(element):
    global median_strategy
    global median_w
    global median_h
    if median_strategy:
        return element.w < median_w and element.h < median_h
    else:
        return element.w < 0.05 and element.h < 0.05

def update_median_size_globals(input_, class_definitions):
    global median_w
    global median_h
    size_list = { "w": [], "h": []}
    for object_class in tqdm.tqdm(range(len(class_definitions))):
        input_class_ = input_ / f"{object_class}"
        if not input_class_.exists():
            raise Exception(f"Error: expected a folder '{str(input_class_)}'")
        for metadata_file_name in sorted(Path(input_class_).resolve().glob("*.json")):
            with open(str(metadata_file_name), "r") as file:
                metadata = json.load(file)
            size_list["w"].append(metadata["w"])
            size_list["h"].append(metadata["h"])
    size_list["w"] = sorted(size_list["w"])
    size_list["h"] = sorted(size_list["h"])
    count = len(size_list["w"])
    if count % 2 == 1:
        count -= 1
    middle_index = int(count / 2)
    median_w = size_list["w"][middle_index]
    median_h = size_list["h"][middle_index]

def read_data(input_dir, patches_per_class):
    element_all = {"large":[], "small":[]}
    input_ = traverse_dir(input_dir)
    class_definitions = None
    with open(str(input_ / "class_definitions.txt"), "r") as file:
        class_definitions = file.read().split("\n")
    if median_strategy:
        update_median_size_globals(input_, class_definitions)
    #for object_class in tqdm.tqdm(range(len(class_definitions))):
    for object_class in range(len(class_definitions)):
        class_counter = 0
        input_class_ = input_ / f"{object_class}"
        if not input_class_.exists():
            raise Exception(f"Error: expected a folder '{str(input_class_)}'")
        selection = list(Path(input_class_).resolve().glob("*.jpg"))
        random.shuffle(selection)
        #for image_file_name in sorted(Path(input_class_).resolve().glob("*.jpg")):
        for image_file_name in selection:
            stem = image_file_name.stem
            metadata_file_name = input_class_ / f"{stem}.json"
            with open(str(metadata_file_name), "r") as file:
                metadata = json.load(file)
            e = element(
                cx=metadata["cx"],
                cy=metadata["cy"],
                h=metadata["h"],
                w=metadata["w"],
                class_id=object_class,
                file_path=str(image_file_name.resolve()))
            if both_dimensions_are_small(e):
                element_all['small'].append(e)
            else:
                element_all['large'].append(e)
            class_counter += 1
            if class_counter >= patches_per_class:
                break
    return element_all

def bestfit_generator(index):
    element_all = read_data(input_dir, patches_per_class)
    if sampling_strategy:
        candidate_num = 500
        large_elements_idx = random.sample(list(range(len(element_all['large']))), int(candidate_num*0.99))
        small_elements_idx = random.sample(list(range(len(element_all['small']))), int(candidate_num*0.01))
    else:
        large_elements_idx = list(range(len(element_all['large'])))
        small_elements_idx = list(range(len(element_all['small'])))
        random.shuffle(large_elements_idx)
        random.shuffle(small_elements_idx)
    large_elements_idx = [element_all['large'][large_idx] for large_idx in large_elements_idx]
    small_elements_idx = [element_all['small'][small_idx] for small_idx in small_elements_idx]
    cand_elements = large_elements_idx + small_elements_idx
    # Initially, randomly put an element
    put_elements = []
    e0 = random.choice(cand_elements)
    cx = random.uniform(min(e0.w/2, 1-e0.w/2), max(e0.w/2, 1-e0.w/2))
    cy = random.uniform(min(e0.h/2, 1-e0.h/2), max(e0.h/2, 1-e0.h/2))
    e0.cx, e0.cy = cx, cy
    put_elements = [e0]
    cand_elements.remove(e0)
    small_cnt = 1 if either_dimension_is_small(e0) else 0
    # Iterativelly insert elements
    while True:
        # Construct meshgrid based on current layout
        put_element_boxes = []
        xticks, yticks = [0,1], [0,1]
        for e in put_elements:
            x1, y1, x2, y2 = e.cx-e.w/2, e.cy-e.h/2, e.cx+e.w/2, e.cy+e.h/2
            xticks.append(x1)
            xticks.append(x2)
            yticks.append(y1)
            yticks.append(y2)
            put_element_boxes.append([x1, y1, x2, y2])
        xticks, yticks = list(set(xticks)), list(set(yticks))
        pticks = list(itertools.product(xticks, yticks))
        meshgrid = list(itertools.product(pticks, pticks))
        put_element_boxes = torch.Tensor(put_element_boxes)
        # Filter out invlid grids
        meshgrid = [grid for grid in meshgrid if grid[0][0] < grid[1][0] and grid[0][1] < grid[1][1]]
        meshgrid_tensor = torch.Tensor([p1 + p2 for p1, p2 in meshgrid])
        iou_res = torchvision.ops.box_iou(meshgrid_tensor, put_element_boxes)
        valid_grid_idx = (iou_res.sum(dim=1) == 0).nonzero().flatten().tolist()
        meshgrid = meshgrid_tensor[valid_grid_idx].tolist()
        # Search for the Mesh-candidate Bestfit pair
        max_fill, max_grid_idx, max_element_idx = 0, -1, -1
        for element_idx, e in enumerate(cand_elements):
            for grid_idx, grid in enumerate(meshgrid):
                if e.w > grid[2] - grid[0] or e.h > grid[3] - grid[1]:
                    continue
                element_area = e.w * e.h
                grid_area = (grid[2] - grid[0]) * (grid[3] - grid[1])
                if element_area/grid_area > max_fill:
                    max_fill = element_area/grid_area
                    max_grid_idx = grid_idx
                    max_element_idx = element_idx
        # Termination condition
        if max_element_idx == -1 or max_grid_idx == -1:
            break
        else:
            maxfit_element = cand_elements[max_element_idx]
            if either_dimension_is_small(maxfit_element):
                small_cnt += 1
            if small_cnt > 5:
                break
            else:
                pass
        # Put the candidate to the center of the grid
        cand_elements.remove(maxfit_element)
        maxfit_element.cx = (meshgrid[max_grid_idx][0] + meshgrid[max_grid_idx][2])/2
        maxfit_element.cy = (meshgrid[max_grid_idx][1] + meshgrid[max_grid_idx][3])/2
        put_elements.append(maxfit_element)
    # Apply a rescale transform to introduce more diversity
    for e in put_elements:
        e.w *= random.uniform(0.8,0.95)
        e.h *= random.uniform(0.8,0.95)
    # Write serialized layouts
    layouts_dir = create_dir(output_dir / "layouts")
    with open(str(Path(layouts_dir) / f"{index}.json"), "w") as f:
        json.dump(put_elements, f, default=lambda x: x.__dict__, indent=2)


median_strategy = False
median_w = 0
median_h = 0
sampling_strategy = False
patches_per_class = 5
layouts_count = 10000
input_dir = "C:\\.patches\\DocBank_40000_patches"

median_suffix = "median" if median_strategy else "percentage"
sampling_suffix = "sampling" if sampling_strategy else "shuffle"
suffix = f"+{median_suffix}+{sampling_suffix}"

output_dir = create_dir(
        create_dir("C:\\.layouts\\") /
        f"DocBank_{layouts_count}{suffix}")

if __name__ == "__main__":
    shutil.copy(Path(input_dir) / "class_definitions.txt", output_dir)
    n_jobs = 8
    with multiprocessing.Pool(n_jobs) as pool:
        pool.starmap(
            bestfit_generator,
            [(i,) for i in range(layouts_count)]
        )
    pool.close()
    pool.join()