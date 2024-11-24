from ultralytics import YOLO

#model = YOLO("yolo11n.yaml") # a new model
#model = YOLO("yolo11n.pt") # a pretrained will be downloaded right now
#model = YOLO("D:\\.dev\\yolo-layout-analysis\\yolo11n.pt") # a local pretrained one
#model = YOLO("D:\\.dev\\yolo-layout-analysis\\yolo11m.pt") # a local pretrained one
model = YOLO("D:\\.dev\\yolo-layout-analysis\\best.pt")
#model = YOLO("D:\\.dev\\yolo-layout-analysis\\last.pt")

if __name__ == '__main__':
    results = model.train(
        #data="D:\\.dev\\yolo-layout-analysis\\datasets\\DocBank\\data.yaml",
        data="D:\\.dev\\yolo-layout-analysis-data\\datasets\\DocBank_bucket_whiteout\\data.yaml",
        #data="C:\\.datasets\\DocBank_bucket_whiteout\\data.yaml",
        #data="C:\\.datasets\\DocSynth300K_split\\data.yaml",
        epochs=10,
        fraction=0.02,
        optimizer="SGD",
        lr0=0.04,
        lrf=0.0001,
    )
