from ultralytics import YOLO
from pathlib import Path
import os

#model = YOLO("best.pt")
#model = YOLO("D:\\.dev\\yolo-layout-analysis\\runs_done\\docbank+median+sampling_yolo11m\\detect\\train\\weights\\best.pt")
#model = YOLO("D:\\.dev\\yolo-layout-analysis\\runs\\detect\\train\\weights\\best.pt")
model = YOLO("D:\\.dev\\yolo-layout-analysis\\runs\\detect\\train2\\weights\\best.pt")

def create_dir(dir):
    dir = Path(f"{dir}")
    if not dir.exists():
        dir.mkdir()
    return dir

#results = model([
#    "C:\\.datasets\\DocBank\\test\\images\\1.tar_1401.0011.gz_dispersion_MNRAS_V2_6_ori.jpg",
#    "C:\\.datasets\\DocBank\\test\\images\\1.tar_1501.00405.gz_Freq-TS-Motifs_2_ori.jpg",
#    "C:\\.datasets\\DocBank\\test\\images\\1.tar_1601.00034.gz_main_4_ori.jpg",
#    "C:\\.datasets\\DocBank\\test\\images\\2.tar_1501.00866.gz_APM-AMC-2015_6_ori.jpg",
#    "C:\\.datasets\\DocBank\\test\\images\\2.tar_1601.00732.gz_curve_lrr_4_ori.jpg",
#])

target_dir = "D:\\Desktop\\pdf-render\\test"
images = [str(Path(target_dir) / image_name) for image_name in os.listdir(target_dir)]
results = model(images)

for (i, result) in enumerate(results):
    #boxes = result.boxes  # Boxes object for bounding box outputs
    #masks = result.masks  # Masks object for segmentation masks outputs
    #keypoints = result.keypoints  # Keypoints object for pose outputs
    #probs = result.probs  # Probs object for classification outputs
    #obb = result.obb  # Oriented boxes object for OBB outputs
    #result.show()  # display to screen
    dir = create_dir("D:\\.dev\\yolo-layout-analysis\\out_yolo")
    filename = str(dir / f"result{i}.jpg")
    result.save(filename=filename)  # save to disk
