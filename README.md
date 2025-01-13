# Land Transportation Vehicle Recognition Application with YOLOv8

This Python-based application uses a Convolutional Neural Network (CNN) model trained with YOLOv8 for vehicle recognition. It provides a user-friendly interface for loading images, running vehicle detection, and viewing results.

---

## Features
- Vehicle detection and classification using a YOLOv8 model.
- Interactive GUI built with Tkinter.
- Easy-to-use interface for loading images and displaying detection results.
- Supports real-time detection using OpenCV.
- Detection include automobile, jeepney, motorcycle, bus, truck, and tricycle

---

## File Structure
VEHICLE-RECOGNITION/
├── model/
│   └── yolov8_model.pt
├── src/
|   ├── app.py
│   └── assets
|         ├── logo.png
│         ├── p.png
|         └── u.png
├── README.md
└── .gitignore

---

## Prerequisites
Ensure you have Python 3.8 or higher installed on your system. 

---

## Installation

### Clone the Repository
```bash
git clone https://github.com/debidabudu/vehicle-recognition.git
cd vehicle-recognition

### Installing Dependencies

Some packages require additional steps for installation. Below are the commands to install them:

1. **OpenCV**:
   ```bash
   pip install opencv-python
   pip install Pillow
   pip install --user ultralytics