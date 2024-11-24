from ultralytics import YOLO
from pathlib import Path

model = YOLO("best.pt")

def create_dir(dir):
    dir = Path(f"{dir}")
    if not dir.exists():
        dir.mkdir()
    return dir

results = model([
    "C:\\.datasets\\DocSynth300K_split\\valid\\images\\1720631300_466168.jpg",
    "C:\\.datasets\\DocSynth300K_split\\valid\\images\\1720630953_7256792.jpg",
    "C:\\.datasets\\DocSynth300K_split\\valid\\images\\1720630677_6501238.jpg",
])

for (i, result) in enumerate(results):
    #boxes = result.boxes  # Boxes object for bounding box outputs
    #masks = result.masks  # Masks object for segmentation masks outputs
    #keypoints = result.keypoints  # Keypoints object for pose outputs
    #probs = result.probs  # Probs object for classification outputs
    #obb = result.obb  # Oriented boxes object for OBB outputs
    #result.show()  # display to screen
    dir = create_dir("D:\\.dev\\yolo-layout-analysis\\prediction")
    filename = str(dir / f"result{i}.jpg")
    result.save(filename=filename)  # save to disk
