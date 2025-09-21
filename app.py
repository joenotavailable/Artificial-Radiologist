from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from ultralytics import YOLO
import os
from werkzeug.utils import secure_filename

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
CORS(app)

model = YOLO(r"/Users/me/AI_testzone/HACKATHON/Bone-Fracture-Detection-main/yolov8n-fracture.pt")
os.makedirs("uploads", exist_ok=True)

@app.route("/", methods=["GET"])
def home():
    return render_template("/index.html")

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         file = request.files['image']
#         file_path = os.path.join("uploads", file.filename)
#         file.save(file_path)

#         results = model.predict(file_path)
#         predictions = results[0].boxes.cls.tolist()

#         os.remove(file_path)  # optional
#         return jsonify({"predictions": predictions})
#     except Exception as e:
#         print("❌ Error during prediction:", e)
#         return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['image']
        filename = secure_filename(file.filename)   # removes spaces, special chars
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        print("💾 Saving uploaded file to:", file_path)
        file.save(file_path)

        print("✅ File exists?", os.path.exists(file_path))

        results = model.predict(file_path)
        detections = results[0].boxes

        predictions = []
        for box in detections:
            predictions.append({
                "x1": float(box.xyxy[0][0]),
                "y1": float(box.xyxy[0][1]),
                "x2": float(box.xyxy[0][2]),
                "y2": float(box.xyxy[0][3]),
                "confidence": float(box.conf[0]),
                "class": int(box.cls[0])
            })

        #os.remove(file_path)  # optional
        return jsonify({"predictions": predictions})
    except Exception as e:
        print("❌ Error during prediction:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000, use_reloader=False)
