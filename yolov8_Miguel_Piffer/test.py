from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor

import cv2

model = YOLO("/Users/miguelpiffer/Desktop/estudos/projeto integrador/runs/detect/train4/weights/best.pt")

results = model.predict(source="/Users/miguelpiffer/Desktop/estudos/projeto integrador/videos/video1.mp4", show = True)
print(results)