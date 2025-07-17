
# ♻️ Waste Classification using MobileNetV2

This project uses **MobileNetV2** for image classification to identify fine-grained waste categories such as plastic, paper, metal, food waste, and more. It serves as the **second stage** in our hybrid AI-based waste management system, enhancing sorting decisions after initial object detection by YOLOv8.

---

## 🛠️ Step-by-Step Guide: Training MobileNetV2 Classifier on Google Colab

### ✅ Step 1: Organize Your Dataset

Prepare your dataset in **ImageFolder format**, where each waste category is stored in a separate folder:

```
dataset/
  ├── Plastic/
  ├── Paper/
  ├── Metal/
  ├── Food_Waste/
  ├── ...
```

Upload this dataset to your Google Drive, for example at:

```
/MyDrive/Wasteclass_mobilenetv2/dataset/
```

---

### ✅ Step 2: Launch Google Colab

Open a new notebook and switch to GPU mode:
**Runtime → Change runtime type → Hardware accelerator → GPU**

---

### ✅ Step 3: Mount Google Drive

Mount your Drive to access the dataset and store model checkpoints:

```python
from google.colab import drive
drive.mount('/content/drive')
```

---

### ✅ Step 4: Install Required Libraries

Google Colab includes most libraries, but ensure compatibility by running:

```python
!pip install torch torchvision matplotlib
```

---

### ✅ Step 5: Load and Preprocess Data

Apply standard transforms like resizing and normalization:

```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
```

Then create the DataLoaders:

```python
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = datasets.ImageFolder(val_dir, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
```

---

### ✅ Step 6: Modify and Train MobileNetV2

Customize the final layer of MobileNetV2 for your dataset:

```python
model = mobilenet_v2(pretrained=False)
model.classifier[1] = nn.Linear(model.last_channel, 12)  # Use your actual class count here
```

Set up the training loop:

```python
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

Run training for the desired number of epochs and track performance.

---

### ✅ Step 7: Save the Trained Model

Once training is complete, save the model checkpoint:

```python
torch.save(model.state_dict(), '/content/drive/MyDrive/Wasteclass_mobilenetv2/mobilenetv2_final.pth')
```

---

### ✅ Step 8: Load Model for Inference (Optional)

To load the trained model later:

```python
model.load_state_dict(torch.load('/content/drive/MyDrive/Wasteclass_mobilenetv2/mobilenetv2_final.pth'))
model.eval()
```

---

## 🚀 Features

* ✅ Lightweight and fast classification with MobileNetV2
* ✅ Trained on 12 custom waste categories
* ✅ Designed to work with YOLOv8 output for two-stage prediction
* ✅ Easily trainable on Google Colab using Google Drive

---

## 📦 Requirements (for Local Setup)

To run the training script locally, install:

```
torch>=1.13.1  
torchvision>=0.14.1  
matplotlib
```

Install using:

```bash
pip install -r requirements.txt
```

---

## 🔗 Dataset

This model was trained on a custom dataset containing **over 30,000 images** spread across 12 distinct waste classes.

