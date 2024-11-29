from ultralytics import YOLO


model_file_name = ".\\.trash\\doclayout_yolo_ft.pt"
#model_file_name = ".\\.trash\\yolo11m_ft.pt"
#model_file_name = ".\\.trash\\yolov10l_ft.pt"


model = YOLO(model_file_name)

print(model)
print("\n")

print("model.names:")
print(model.names)
