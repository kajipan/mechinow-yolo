from flask import Flask, request, jsonify
from ultralytics import YOLO
import os

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "bike_body_type.pt")
model = YOLO(MODEL_PATH)

CLASS_NAME_MAP = {
    "scooter": "scooter",
    "commuter_standard": "commuter_standard",
    "sports_commuter": "sports_commuter",
}

DEFAULT_BODY_TYPE = "commuter_standard"


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "no image field in request"}), 400

    image_file = request.files["image"]
    temp_path = os.path.join(os.path.dirname(__file__), "_temp_upload.jpg")
    image_file.save(temp_path)

    try:
        results = model(temp_path)
        result = results[0]

        if hasattr(result, "probs") and result.probs is not None:
            top_index = int(result.probs.top1)
            confidence = float(result.probs.top1conf)
            raw_class_name = model.names[top_index]
        else:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                return jsonify({"bodyType": DEFAULT_BODY_TYPE, "confidence": 0.0})
            best = boxes[boxes.conf.argmax()]
            raw_class_name = model.names[int(best.cls[0])]
            confidence = float(best.conf[0])

        body_type = CLASS_NAME_MAP.get(raw_class_name, DEFAULT_BODY_TYPE)

        return jsonify({
            "bodyType": body_type,
            "rawClassName": raw_class_name,
            "confidence": round(confidence, 3),
        })
    except Exception as e:
        return jsonify({"error": str(e), "bodyType": DEFAULT_BODY_TYPE}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "classes": model.names})


if __name__ == "__main__":
    print("Model classes loaded:", model.names)
    app.run(host="0.0.0.0", port=5000)