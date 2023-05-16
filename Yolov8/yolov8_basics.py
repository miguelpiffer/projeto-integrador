from ultralytics import YOLO
import numpy

#carregar o model;o do yolo desejado
model = YOLO("yolov8n.pt","v8")

detection_output= model.predict(source="/Users/miguelpiffer/Desktop/estudos/projeto integrador/images ", conf=0.25,save=False)

print( detection_output)

print(detection_output[0].numpy())


