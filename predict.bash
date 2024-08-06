#!/bin/bash

# Prompt the user for input
read -p "Image name: " imgName

yolo predict model=runs/detect/train8/weights/best.pt source="./../human-vehicles.v1-human-vehicles-zhivanai.yolov8/test/images/${imgName}.jpg" imgsz=640