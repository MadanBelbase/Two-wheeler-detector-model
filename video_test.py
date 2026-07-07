from ultralytics import YOLO

# Load your trained model
model = YOLO("model/best.pt")

# Run prediction on video
results = model.predict(
    source="test_videos/traffic.mp4",
    conf=0.25,
    save=True,
    project="outputs",
    name="video",
    exist_ok=True
)

print("Detection completed!")
print("output saved to: outputs/video/")