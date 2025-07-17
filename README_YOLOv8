
# ♻️ Waste-Classification

This project implements a deep learning-based waste classification system using YOLOv8 for object detection and Streamlit for an interactive web interface. The system can detect and classify various waste types (such as glass, paper, metal, cardboard, etc.) in images, videos, webcam streams, and even YouTube links. It aims to promote smarter recycling through real-time waste segregation.

---

## 🚀 Features

* ⚡ Real-time waste detection
* 🧠 YOLOv8 object detection
* 🌐 Streamlit-based web app
* 📹 Supports image, video, webcam, and YouTube input
* 🛰️ Object tracking with ByteTrack and BoT-SORT
* 💡 Modular structure for easy expansion

---

## 📁 Dataset

The model is trained using the garbage classification dataset available at:
[Roboflow Garbage Dataset](https://universe.roboflow.com/project-1ycjd/garbage-classification-jxps3/dataset)

---

## 🛠️ Setup Instructions

### ✅ Step 1: Clone the Repository

```bash
git clone https://github.com/devashreesaran/Waste_Management.git
cd Waste_Management
```

### ✅ Step 2: (Optional) Create a Virtual Environment

```bash
python -m venv env
env\Scripts\activate  # For Windows
# OR
source env/bin/activate  # For macOS/Linux
```

### ✅ Step 3: Install Dependencies

Install packages using:

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install ultralytics streamlit opencv-python pafy youtube-dl
```

### ✅ Step 4: Add the Trained Model

Place your trained YOLOv8 weights (e.g., `best.pt`) in:

```
streamlit-detection-tracking-app/weights/yolov8_custom.pt
```

Update the model path in `helper.py`:

```python
model = YOLO("path/to/yolov8_custom.pt")
```

### ✅ Step 5: Run the Application

Navigate to the app folder and launch Streamlit:

```bash
cd streamlit-detection-tracking-app
streamlit run app.py
```

---

## 🖼️ How to Use

* Upload images/videos or paste a YouTube URL.
* Optionally use webcam detection.
* Toggle object tracking features.
* View the output with bounding boxes and labels.

---

## 🧪 Training Your Own Model (Optional)

If you'd like to retrain:

```bash
yolo task=detect mode=train model=yolov8n.pt data=your_data.yaml epochs=50 imgsz=640
```

Replace the weight file path in your code after training.


