from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
import joblib
import pandas as pd
from ultralytics import YOLO
import os
import uuid
from werkzeug.utils import secure_filename
import logging
import gzip

# ============================
# Configuration & Setup
# ============================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# CORS configuration – must be right after app init
CORS(
    app,
    origins=["*"],
    supports_credentials=True
)

# App configuration
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}
UPLOAD_FOLDER = "static/uploads"
RESULTS_FOLDER = "static/results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# ============================
# Model Loading
# ============================
try:
    logger.info("Loading AI models...")

    damage_model = YOLO("Final_Damage/DamageTypebest.pt")
    severity_model = YOLO("Final_Damage/Severitybest.pt")

    with gzip.open("Final_Damage/cost_model.pkl.gz", "rb") as f:
        cost_model = joblib.load(f)

    encoders = joblib.load("Final_Damage/label_encoders.pkl")
    feature_cols = joblib.load("Final_Damage/feature_columns.pkl")

    logger.info("All models loaded successfully.")
except Exception as e:
    logger.error(f"Error loading models: {e}")
    raise e

# ============================
# Safe Label Encoder
# ============================
class SafeLabelEncoder:
    def __init__(self, le):
        self.le = le
        self.classes = set(le.classes_)

    def transform(self, values):
        return self.le.transform(
            [v if v in self.classes else "__UNKNOWN__" for v in values]
        )

for col in encoders:
    le = encoders[col]
    le.classes_ = np.append(le.classes_, "__UNKNOWN__")
    encoders[col] = SafeLabelEncoder(le)

# ============================
# Helpers
# ============================
def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )

def save_image(file):
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
        file.save(filepath)
        return filepath
    return None

def analyze_damage(img_path):
    try:
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError("Could not read image file")

        h, w, _ = img.shape
        total_area = h * w

        d_pred = damage_model(img)
        s_pred = severity_model(img)

        if len(d_pred[0].boxes) == 0:
            logger.info("No damage detected in the image")
            return []

        results = []
        for box in d_pred[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cls_id = int(box.cls[0])
            damage_name = damage_model.names[cls_id]
            confidence = float(box.conf[0])

            area_ratio = ((x2 - x1) * (y2 - y1)) / total_area

            severity_label = "minor"
            if len(s_pred[0].boxes) > 0:
                s_box = max(s_pred[0].boxes, key=lambda x: float(x.conf[0]))
                severity_label = severity_model.names[int(s_box.cls[0])]

            results.append(
                {
                    "damage_type": damage_name,
                    "severity": severity_label,
                    "damage_area_ratio": float(area_ratio),
                    "confidence": confidence,
                    "box": [int(x1), int(y1), int(x2), int(y2)],
                }
            )

        logger.info(f"Detected {len(results)} damage areas")
        return results
    except Exception as e:
        logger.error(f"Error in damage analysis: {e}")
        return []

def estimate_cost(damage_info_list, user_vehicle):
    try:
        rows = []
        for d in damage_info_list:
            rows.append(
                {
                    "damage_type": d["damage_type"],
                    "severity": d["severity"],
                    "damage_area_ratio": d["damage_area_ratio"],
                    "primary_damage": d["damage_type"],
                    "brand": user_vehicle["brand"],
                    "model": user_vehicle["model"],
                    "year": user_vehicle["year"],
                    "fuel": user_vehicle["fuel"],
                    "type": user_vehicle["type"],
                    "color": user_vehicle["color"],
                }
            )

        df = pd.DataFrame(rows)

        df = df.replace("", "__UNKNOWN__")
        df = df.fillna("__UNKNOWN__")

        for col, encoder in encoders.items():
            if col in df.columns:
                df[col] = encoder.transform(df[col].astype(str))

        df = df.reindex(columns=feature_cols, fill_value=0)

        cost = float(cost_model.predict(df)[0])
        return max(1000.0, cost)
    except Exception as e:
        logger.error(f"Error in cost estimation: {e}")
        return 5000.0

# ============================
# Routes
# ============================
@app.route("/")
def home():
    return jsonify(
        {
            "message": "Car Damage Detection API",
            "status": "running",
            "version": "1.0",
            "endpoints": ["/health", "/predict", "/vehicle-brands"],
        }
    )

@app.route("/health")
def health():
    return jsonify(
        {
            "status": "healthy",
            "models_loaded": True,
            "service": "Car Damage Detection API",
        }
    )

@app.route("/vehicle-brands")
def get_vehicle_brands():
    brands_models = {
        "Toyota": ["Fortuner", "Innova", "Glanza", "Camry", "Corolla"],
        "Hyundai": ["Creta", "Venue", "i20", "i10", "Verna"],
        "Tata": ["Harrier", "Nexon", "Punch", "Safari", "Tiago"],
        "Honda": ["City", "Civic", "Amaze", "Accord", "CR-V"],
        "Kia": ["Seltos", "Sonet", "Carens", "Carnival"],
        "Mahindra": ["XUV500", "Scorpio", "Bolero", "Thar"],
        "Ford": ["Ecosport", "Endeavour", "Figo", "Aspire"],
        "MarutiSuzuki": ["Swift", "Baleno", "Dzire", "WagonR"],
    }
    return jsonify(brands_models)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        img_path = save_image(file)
        if not img_path:
            return jsonify(
                {"error": "Invalid file type. Allowed: png, jpg, jpeg, gif"}
            ), 400

        logger.info(f"Processing image: {img_path}")

        user_vehicle = {
            "brand": request.form.get("brand", "Toyota"),
            "model": request.form.get("model", "Fortuner"),
            "year": int(request.form.get("year", 2020)),
            "fuel": request.form.get("fuel", "Petrol"),
            "type": request.form.get("type", "SUV"),
            "color": request.form.get("color", "White"),
        }

        logger.info(f"Vehicle details: {user_vehicle}")

        damage_results = analyze_damage(img_path)
        if not damage_results:
            return jsonify(
                {
                    "success": True,
                    "message": "No damage detected in the image",
                    "damage_count": 0,
                    "total_cost": 0,
                    "cost_results": [],
                    "crops": [],
                    "annotated_image": None,
                    "vehicle_info": user_vehicle,
                    "currency": "INR",
                }
            )

        annotated_image = cv2.imread(img_path)
        annotated_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

        cropped_images = []
        cost_results = []
        total_cost = 0.0

        for damage in damage_results:
            x1, y1, x2, y2 = damage["box"]

            label = f"{damage['damage_type']} ({damage['severity']})"
            cv2.rectangle(
                annotated_rgb, (x1, y1), (x2, y2), (255, 0, 0), 3
            )
            cv2.putText(
                annotated_rgb,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            crop = annotated_image[y1:y2, x1:x2]
            if crop.size > 0:
                crop_filename = f"{uuid.uuid4()}_crop.jpg"
                crop_path = os.path.join(RESULTS_FOLDER, crop_filename)
                cv2.imwrite(crop_path, crop)
                cropped_images.append(f"static/results/{crop_filename}")

        total_cost = estimate_cost(damage_results, user_vehicle)

        for damage in damage_results:
            cost_results.append(
                {
                    "damage_type": damage["damage_type"],
                    "severity": damage["severity"],
                    "confidence": round(damage["confidence"], 2),
                    "cost": round(total_cost / len(damage_results), 2),
                }
            )

        annotated_filename = f"{uuid.uuid4()}_annotated.jpg"
        annotated_path = os.path.join(RESULTS_FOLDER, annotated_filename)
        cv2.imwrite(
            annotated_path,
            cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR),
        )

        response = {
            "success": True,
            "damage_count": len(damage_results),
            "total_cost": round(total_cost, 2),
            "cost_results": cost_results,
            "crops": cropped_images,
            "annotated_image": f"static/results/{annotated_filename}",
            "vehicle_info": user_vehicle,
            "currency": "INR",
        }
        logger.info(
            f"Prediction completed: {len(damage_results)} damages, total cost {total_cost:.2f}"
        )
        return jsonify(response)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return (
            jsonify(
                {
                    "success": False,
                    "error": str(e),
                    "message": "Internal server error during prediction",
                }
            ),
            500,
        )

# ============================
# Static file routes
# ============================
@app.route("/static/<path:path>")
def serve_static(path):
    return send_from_directory("static", path)

@app.route("/static/uploads/<path:filename>")
def serve_upload(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route("/static/results/<path:filename>")
def serve_result(filename):
    return send_from_directory(RESULTS_FOLDER, filename)

# ============================
# Error handlers
# ============================
@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Maximum size is 16MB"}), 413

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error"}), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed"}), 405

# ============================
# Entry point
# ============================
if __name__ == "__main__":
    logger.info("Starting Car Damage Detection API Server...")
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True, threaded=True)
