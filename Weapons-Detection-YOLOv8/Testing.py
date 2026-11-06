from ultralytics import YOLO

# Load the trained model (update path if needed)
model = YOLO("best.pt")
results = model.predict(
    source="R.jpg",  # path to your test image
    conf=0.5,   # confidence threshold (adjust if needed)
    save=True   # saves the annotated image automatically
)