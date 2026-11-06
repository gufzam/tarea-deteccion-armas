# Weapons-Detection-YOLOv8
This project detects **guns** in images using **YOLOv8** and a custom dataset from Roboflow.

---

## 📂 Files in this Repository

- `weapons-detection.ipynb` – Jupyter notebook for training the YOLOv8 model.  
- `Testing.py` – Python script for running inference on images or videos.  
- `OIP.jpg` – Example image for testing predictions (optional).  
- `best.pt` – Trained YOLOv8 model weights.  
- `runs/` – Training logs and saved predictions (ignored in GitHub).  

---

## ⚡ Requirements

```bash
pip install ultralytics roboflow opencv-python matplotlib supervision 
