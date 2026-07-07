from app.detector import detect, model

results = detect("test_images/image.png")

for box in results[0].boxes:
    cls = int(box.cls)

    print(f"Class: {model.names[cls]}")
    print(f"Confidence: {float(box.conf):.2f}")
    print(f"Bounding Box: {box.xyxy.tolist()}")
    print("-" * 40)

print("\nResult saved in: outputs/predict/")