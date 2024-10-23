from ultralytics import YOLO


# import torch
# print(torch.cuda.is_available())
# print(torch.backends.mps.is_available())


# model = YOLO("yolov8x")
# model = YOLO("models/best.pt")
model = YOLO("models/best4.pt")


results = model(source="videos/football.mp4", show=False, save=True, conf=0.25)
print(results[0])
print('=========================================')