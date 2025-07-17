import streamlit as st
st.set_page_config(page_title="Waste Detector", layout="wide")

from ultralytics import YOLO
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import os
import cv2
import numpy as np
import tempfile
from torchvision.models import mobilenet_v2

# -----------------------------
# Load Models (YOLO + MobileNet)
# -----------------------------
@st.cache_resource
def load_models():
    yolo_model = YOLO("C:/Users/mudit/Final_project/best.pt")
    mobilenet_model = mobilenet_v2(pretrained=False)
    classes = [
        'aerosol_cans', 'food_waste', 'general', 'glass_bottlles', 'glass_jars',
        'metal_cans', 'paper_cups', 'plastic_trash_bags', 'plastic_water_bottles',
        'plastic_cup_lids', 'magzines', 'newspaper'
    ]
    mobilenet_model.classifier[1] = torch.nn.Linear(mobilenet_model.last_channel, len(classes))
    mobilenet_model.load_state_dict(torch.load("C:/Users/mudit/Final_project/mobilenetv2_fine_classifier.pth", map_location='cpu'))
    mobilenet_model.eval()
    return yolo_model, mobilenet_model, classes

yolo_model, mobilenet_model, classes = load_models()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

coarse_to_fine = {
    "PLASTIC": ['plastic_trash_bags', 'plastic_water_bottles', 'plastic_cup_lids'],
    "METAL": ['aerosol_cans', 'metal_cans'],
    "PAPER": ['paper_cups', 'magzines', 'newspaper'],
    "GLASS": ['glass_bottlles', 'glass_jars'],
    "BIODEGRADABLE": ['food_waste'],
    "CARDBOARD": ['general'],
}

# -----------------------------
# UI and Upload
# -----------------------------
st.title("♻️ Waste Detection and Subclassification")
uploaded_file = st.file_uploader("📂 Upload an image", type=["jpg", "jpeg", "png"])

def classify_with_mobilenet_filtered(crop_img_path, yolo_label):
    image = Image.open(crop_img_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = mobilenet_model(input_tensor)
        probs = F.softmax(output, dim=1)[0].cpu()

    if yolo_label.upper() in coarse_to_fine:
        fine_classes = coarse_to_fine[yolo_label.upper()]
        fine_indices = [classes.index(cls) for cls in fine_classes]
        filtered_probs = probs[fine_indices]
        top_idx = torch.argmax(filtered_probs).item()
        predicted_class = fine_classes[top_idx]
        confidence = filtered_probs[top_idx].item()

        return predicted_class, confidence
    else:
        return "Unknown", 0.0

# -----------------------------
# Detection and Display
# -----------------------------
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # Run YOLO on image path (safer and cleaner)
    results = yolo_model(tmp_path)
    image = Image.open(tmp_path).convert("RGB")
    img_np = np.array(image)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    st.image(image, caption="📸 Uploaded Image", use_container_width=True)

    boxes = results[0].boxes
    annotated_img = img_bgr.copy()

    st.markdown("### 🔍 Detection and Subclassification Results")
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        class_name = yolo_model.names[int(box.cls[0])]
        conf = float(box.conf[0])

        crop = img_bgr[y1:y2, x1:x2]
        crop_path = f"crop_{i}_{class_name}.jpg"
        cv2.imwrite(crop_path, crop)

        subclass, subclass_conf = classify_with_mobilenet_filtered(crop_path, class_name)

        label = f"{class_name} → {subclass} ({subclass_conf:.2f})"
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated_img, label, (x1, max(y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        st.markdown(f"✅ **Detection {i+1}** — `{class_name}` → `{subclass}` **(Conf: {subclass_conf:.2f})**")

    st.image(annotated_img, caption="🧾 Annotated Image with Subclasses", channels="BGR", use_container_width=True)



        
