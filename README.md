🔥 Fire Detection Using YOLOv8

Real-Time Fire Detection Model for CCTV, Indoor Safety Monitoring, and Early Hazard Warning Systems

📌 Overview

This project implements a YOLOv8-based fire detection system trained on a custom dataset of 3500 fire and background images.
The goal is to detect fire in:

Indoor environments (homes, buildings, stadiums)

CCTV-like stable camera setups

Real-time surveillance systems

The model is trained using Ultralytics YOLOv8 and deployed using Gradio.

🚀 Features

✔ Real-time fire detection
✔ Trained on a custom dataset
✔ High recall on fire class
✔ YOLOv8 small model (fast + accurate)
✔ Includes Gradio Web UI for deployment
✔ Can run on CPU or GPU

📂 Project Structure
📦 Fire-Detection-YOLOv8
├── best.pt                   # Trained fire detection model
├── app.py                    # Gradio deployment script
├── data.yaml                 # Dataset configuration
├── requirements.txt          # Project dependencies
├── README.md                 # Documentation
├── /dataset/                 # (Optional) Only if you want to include sample images
│   ├── train/
│   ├── valid/
│   └── test/
└── /docs/                    # Architecture diagrams, training plots

🧠 YOLOv8 Architecture (Brief)

YOLOv8 consists of:

1️⃣ Backbone (Feature Extraction)

Uses C2f blocks

Learns edges, flames, smoke textures

2️⃣ Neck (Feature Fusion)

PAN (Path Aggregation Network)

Combines low-level and high-level features

3️⃣ Detection Head

Predicts:

Bounding boxes

Object class (fire / background)

Confidence scores

This architecture helps YOLOv8 detect fire at multiple scales.

🛠 Installation
1️⃣ Clone the repository
git clone https://github.com/your-username/Fire-Detection-YOLOv8.git
cd Fire-Detection-YOLOv8

2️⃣ Install requirements
pip install -r requirements.txt

🎯 Model Training Command

This is the final training command used:

yolo detect train \
  data="/content/Firee_detection_dataset/Fire dataset YOLOV8/data.yaml" \
  model=yolov8s.pt \
  epochs=120 \
  imgsz=650 \
  batch=16 \
  cache=True \
  amp=True \
  hsv_h=0.02 hsv_s=0.7 hsv_v=0.4 \
  degrees=10 translate=0.1 scale=0.8 fliplr=0.5 mosaic=0.8 \
  patience=20 \
  name=Fire_detector_fast

📊 Model Performance

Confusion Matrix (from training results):

Fire Detection Accuracy: Good

Recall: High → catches most fire cases

Background Accuracy: Reasonable but misclassifications exist

mAP50: ~45%

mAP50-95: ~16%

The model works well for real-time indoor fire detection but can be improved with more data.

🎥 Run the Gradio App (Deployment)
1️⃣ Add your best.pt file

Place the model in the root folder.

2️⃣ Run the app
python app.py

3️⃣ Gradio Interface

Upload video

Detect fire in real-time

Results saved automatically

🧪 Example Inference Code
from ultralytics import YOLO

model = YOLO("best.pt")
model.predict("sample_video.mp4", save=True, conf=0.4)

🌐 Hosting Options
Platform	Speed	Free	Recommended
Hugging Face Spaces	⭐⭐⭐	Yes	✔ Best overall
Render	⭐⭐⭐⭐	Limited	Good
Roboflow Inference	⭐⭐⭐⭐⭐	No	Paid, very fast
Google Colab + Ngrok	⭐⭐⭐	Free	Temporary
✔ Advantages

Real-time detection

Lightweight model

Runs on CPU

Good for CCTV-based fire alerts

❌ Limitations

May detect bright lights as fire

Low-light performance not perfect

Needs diverse training data
