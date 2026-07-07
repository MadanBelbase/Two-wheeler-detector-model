from ultralytics import YOLO

MODEL_PATH = "model/best.pt"

model = YOLO(MODEL_PATH)

def detect(image_path):
    return model.predict(
        source=image_path,
        conf=0.25,
        save=True,        
        project="outputs", 
        name="predict",  
        exist_ok=True       
    )