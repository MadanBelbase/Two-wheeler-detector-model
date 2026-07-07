# 🚗 Two-Wheeler Detection using YOLOv8

A deep learning-based object detection system for detecting and classifying vehicles using **YOLOv8**. The model is trained on a custom vehicle dataset and can detect multiple vehicle types from images, videos, and live webcam feeds.

---

## 📌 Features

- 🚀 Custom-trained YOLOv8 model
- 🏍️ Detects multiple vehicle classes
- 🖼️ Image detection
- 🎥 Video detection
- 📷 Webcam detection
- 📦 Fast inference using Ultralytics YOLO
- 🌐 Ready for FastAPI web deployment
- 💾 Save annotated detection results

---

## 📂 Project Structure

```text
Two-wheeler-detector-model/
│
├── app/
│   ├── detector.py
│   ├── __init__.py
│
├── datasets/
│   ├── classification/
│   └── data.json
│
├── model/
│   └── best.pt
│
├── notebook/
│   └── vehicle_detect.ipynb
│
├── outputs/
│
├── test_images/
│   └── test.jpg
│
├── test.py
├── requirements.txt
└── README.md
```

---

## 🚘 Vehicle Classes

The trained model detects the following vehicle classes:

| Class ID | Vehicle Type |
|----------|--------------|
| 0 | Auto Rickshaw |
| 1 | Bicycle |
| 2 | Bus |
| 3 | Car |
| 4 | Motorcycle |
| 5 | Person |
| 6 | Truck |

---

## 🛠️ Tech Stack

- Python
- YOLOv8 (Ultralytics)
- OpenCV
- NumPy
- FastAPI (Deployment)
- Jupyter Notebook

---

## ⚙️ Installation

### Clone the repository

```bash
git clone https://github.com/<your-username>/Two-wheeler-detector-model.git
```

```bash
cd Two-wheeler-detector-model
```

---

### Create a virtual environment

```bash
python -m venv venv
```

Windows

```bash
venv\Scripts\activate
```

macOS/Linux

```bash
source venv/bin/activate
```

---

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Run Image Detection

```bash
python test.py
```

Example Output

```
Class: motorcycle
Confidence: 0.94

Class: truck
Confidence: 0.91

Class: car
Confidence: 0.89
```

The annotated image will be saved inside:

```
outputs/
```

---

## 🖼️ Detect a Custom Image

Replace the image path inside `test.py`:

```python
results = detect("test_images/your_image.jpg")
```

Run:

```bash
python test.py
```

---

## 🎥 Video Detection

```python
from ultralytics import YOLO

model = YOLO("model/best.pt")

model.predict(
    source="video.mp4",
    save=True
)
```

---

## 📷 Webcam Detection

```python
from ultralytics import YOLO

model = YOLO("model/best.pt")

model.predict(
    source=0,
    show=True
)
```

---

## 🌐 Future Improvements

- FastAPI REST API
- Modern Web Interface
- Drag-and-Drop Image Upload
- Video Upload Support
- Live Camera Detection
- Docker Deployment
- Cloud Deployment (AWS, Azure, or GCP)

---

## 📈 Model Capabilities

- Multi-class Object Detection
- Bounding Box Prediction
- Confidence Score Estimation
- Real-time Inference
- Custom Dataset Support

---

## 📚 Dataset

The model is trained on a custom dataset containing multiple vehicle categories for object detection.

---

## 👨‍💻 Author

**Madan Belbase**

GitHub: https://github.com/MadanBelbase

---

## 📄 License

This project is intended for educational and research purposes.

---

⭐ If you find this project useful, consider giving it a star!
