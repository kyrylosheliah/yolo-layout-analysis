from ultralytics import YOLO

#model = YOLO("yolo11l.yaml")
#model = YOLO("yolo11n.yaml")
#model = YOLO("yolo11m.yaml")
#model = YOLO("yolo11n.pt")
#model = YOLO("D:\\.dev\\yolo-layout-analysis\\best.pt")
#model = YOLO("D:\\.dev\\yolo-layout-analysis\\best_pretrain_2_median_sampling.pt")
model = YOLO("D:\\.dev\\yolo-layout-analysis\\best_pretrain_2_median_shuffle.pt")

if __name__ == '__main__':
    results = model.train(
        #data="D:\\.dev\\yolo-layout-analysis\\datasets\\DocBank\\data.yaml",
        #data="D:\\.dev\\yolo-layout-analysis-data\\datasets\\DocBank_bucket_whiteout\\data.yaml",
        #data="C:\\.datasets\\DocBank_bucket_whiteout\\data.yaml",
        #data="C:\\.datasets\\DocSynth300K_split\\data.yaml",
        #data="C:\\.datasets_converted\\DocBank_10000+percentage+shuffle_split\\data.yaml",
        #data="C:\\.datasets_converted\\DocBank_10000+median+shuffle_split\\data.yaml",
        #data="C:\\.datasets_converted\\DocBank_10000+percentage+sampling_split\\data.yaml",
        #data="C:\\.datasets_converted\\DocBank_10000+percentage+sampling_1000_split\\data.yaml",
        #data="C:\\.datasets_converted\\DocBank_10000+median+sampling_split\\data.yaml",
        #data="C:\\.datasets\\DocBank\\data.yaml",
        data="C:\\.datasets\\custom_rearrange\\data.yaml",
        epochs=200,
        #fraction=1,
        #fraction=0.04,
        optimizer="SGD",
        #lr0=0.04,
        lr0=0.002,
        lrf=0.0001,
        #cache=False,
        #batch=16,
        #imgsz=1000,
    )
