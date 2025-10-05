from ultralytics import YOLO
import cv2
import hashlib
import os

model = YOLO("yolov8n.pt")

def detect_ppe(image_path: str):
    results = model(image_path)[0]
    bboxes = []
    helmet = False
    vest = False
    img = cv2.imread(image_path)
    h, w, _ = img.shape

    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        label_id = int(box.cls[0])
        label = results.names[label_id]
        if label not in ["helmet", "vest"]:
            continue
        if label == "helmet":
            helmet = True
        elif label == "vest":
            vest = True
        bboxes.append({
            "label": label,
            "x": x1/w,
            "y": y1/h,
            "w": (x2-x1)/w,
            "h": (y2-y1)/h
        })
        color = (0,255,0) if label=="helmet" else (0,0,255)
        cv2.rectangle(img, (int(x1),int(y1)), (int(x2),int(y2)), color, 2)
        cv2.putText(img, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    OVERLAY_FOLDER = os.path.join(BASE_DIR, "static", "overlays")
    os.makedirs(OVERLAY_FOLDER, exist_ok=True)
    overlay_path = os.path.join(OVERLAY_FOLDER, os.path.basename(image_path))
    cv2.imwrite(overlay_path, img)

    with open(image_path, "rb") as f:
        detections_hash = hashlib.md5(f.read()).hexdigest()

    return {
        "helmet": helmet,
        "vest": vest,
        "bboxes": bboxes,
        "detections_hash": detections_hash,
        "overlay_url": f"/static/overlays/{os.path.basename(image_path)}"
    }
