# from flask import Flask, request, jsonify, send_from_directory
# from flask_cors import CORS
# import cv2
# import numpy as np
# import joblib
# import pandas as pd
# from ultralytics import YOLO
# import os
# import uuid
# from werkzeug.utils import secure_filename
# import logging
# import gzip
# import time
# import psutil
# from threading import Thread
# import gc

# # ============================
# # Configuration & Setup
# # ============================
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# app = Flask(__name__)

# # CORS configuration
# CORS(
#     app,
#     origins=["https://vd-dlproject.vercel.app/", "*"],
#     supports_credentials=True,
#     allow_headers=["Content-Type", "Authorization"],
#     methods=["GET", "POST", "OPTIONS"]
# )

# # Optimized configuration for Render free tier
# app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # Reduced to 8MB
# ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
# UPLOAD_FOLDER = "static/uploads"
# RESULTS_FOLDER = "static/results"

# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(RESULTS_FOLDER, exist_ok=True)

# # ============================
# # Global Variables for Model Management
# # ============================
# damage_model = None
# severity_model = None
# cost_model = None
# encoders = None
# feature_cols = None
# models_loaded = False

# # ============================
# # Memory and Performance Monitoring
# # ============================
# def get_memory_usage():
#     """Get current memory usage in MB"""
#     process = psutil.Process(os.getpid())
#     return process.memory_info().rss / 1024 / 1024

# def cleanup_old_files():
#     """Clean up old uploaded and result files to save space"""
#     try:
#         current_time = time.time()
#         for folder in [UPLOAD_FOLDER, RESULTS_FOLDER]:
#             if os.path.exists(folder):
#                 for filename in os.listdir(folder):
#                     filepath = os.path.join(folder, filename)
#                     if os.path.isfile(filepath):
#                         # Remove files older than 1 hour
#                         if current_time - os.path.getmtime(filepath) > 3600:
#                             os.remove(filepath)
#                             logger.info(f"Cleaned up old file: {filename}")
#     except Exception as e:
#         logger.error(f"Error cleaning up files: {e}")

# # ============================
# # Optimized Model Loading
# # ============================
# def load_models():
#     """Load models with memory optimization"""
#     global damage_model, severity_model, cost_model, encoders, feature_cols, models_loaded

#     if models_loaded:
#         return True

#     try:
#         logger.info("Loading AI models with memory optimization...")

#         # Force garbage collection before loading
#         gc.collect()

#         # Check available memory before loading
#         available_memory = psutil.virtual_memory().available / 1024 / 1024
#         logger.info(f"Available memory: {available_memory:.1f} MB")

#         if available_memory < 300:  # Less than 300MB available
#             logger.warning("Low memory detected, attempting cleanup...")
#             cleanup_old_files()
#             gc.collect()

#         # Load YOLO models with optimization
#         damage_model = YOLO("../Backend/Final_Damage/DamageTypebest.pt")
#         severity_model = YOLO("../Backend/Final_Damage/Severitybest.pt")

#         # Load cost model
#         with gzip.open("../Backend/Final_Damage/cost_model.pkl.gz", "rb") as f:
#             cost_model = joblib.load(f)

#         encoders = joblib.load("../Backend/Final_Damage/label_encoders.pkl")
#         feature_cols = joblib.load("../Backend/Final_Damage/feature_columns.pkl")

#         models_loaded = True
#         logger.info("All models loaded successfully.")
#         logger.info(f"Memory usage after loading: {get_memory_usage():.1f} MB")
#         return True

#     except Exception as e:
#         logger.error(f"Error loading models: {e}")
#         return False

# # ============================
# # Safe Label Encoder
# # ============================
# class SafeLabelEncoder:
#     def __init__(self, le):
#         self.le = le
#         self.classes = set(le.classes_)

#     def transform(self, values):
#         return self.le.transform(
#             [v if v in self.classes else "__UNKNOWN__" for v in values]
#         )

# # ============================
# # Optimized Helpers
# # ============================
# def allowed_file(filename):
#     return (
#         "." in filename
#         and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
#     )

# def save_image(file):
#     """Save image with size validation"""
#     if file and allowed_file(file.filename):
#         # Check file size
#         file.seek(0, os.SEEK_END)
#         size = file.tell()
#         file.seek(0)

#         if size > 8 * 1024 * 1024:  # 8MB limit
#             return None

#         filename = secure_filename(file.filename)
#         unique_filename = f"{uuid.uuid4()}_{filename}"
#         filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
#         file.save(filepath)
#         return filepath
#     return None

# def optimize_image(img_path, max_size=(640, 640)):
#     """Optimize image for faster processing"""
#     try:
#         img = cv2.imread(img_path)
#         if img is None:
#             return None

#         h, w = img.shape[:2]

#         # Resize if too large
#         if w > max_size[0] or h > max_size[1]:
#             ratio = min(max_size[0] / w, max_size[1] / h)
#             new_w, new_h = int(w * ratio), int(h * ratio)
#             img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

#         # Save optimized image
#         cv2.imwrite(img_path, img, [cv2.IMWRITE_JPEG_QUALITY, 85])
#         return img_path

#     except Exception as e:
#         logger.error(f"Error optimizing image: {e}")
#         return img_path

# def analyze_damage_optimized(img_path):
#     """Optimized damage analysis with timeout and memory management"""
#     start_time = time.time()

#     try:
#         # Optimize image first
#         img_path = optimize_image(img_path)
#         if not img_path:
#             return []

#         img = cv2.imread(img_path)
#         if img is None:
#             raise ValueError("Could not read image file")

#         h, w, _ = img.shape
#         total_area = h * w

#         # Run damage detection with confidence threshold
#         d_pred = damage_model(img, conf=0.3, verbose=False)

#         if len(d_pred[0].boxes) == 0:
#             logger.info("No damage detected in the image")
#             return []

#         # Process only top 3 damage detections to save time
#         boxes = d_pred[0].boxes[:3]  # Limit to top 3

#         results = []
#         for box in boxes:
#             x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
#             cls_id = int(box.cls[0])
#             damage_name = damage_model.names[cls_id]
#             confidence = float(box.conf[0])

#             area_ratio = ((x2 - x1) * (y2 - y1)) / total_area

#             # Simplified severity detection - use confidence as proxy
#             severity_label = "minor"
#             if confidence > 0.7:
#                 severity_label = "moderate"
#             elif confidence > 0.8:
#                 severity_label = "high"

#             results.append({
#                 "damage_type": damage_name,
#                 "severity": severity_label,
#                 "damage_area_ratio": float(area_ratio),
#                 "confidence": confidence,
#                 "box": [int(x1), int(y1), int(x2), int(y2)],
#             })

#         processing_time = time.time() - start_time
#         logger.info(f"Damage analysis completed in {processing_time:.2f}s: {len(results)} areas detected")
#         return results

#     except Exception as e:
#         logger.error(f"Error in damage analysis: {e}")
#         return []

# def estimate_cost_optimized(damage_info_list, user_vehicle):
#     """Optimized cost estimation"""
#     try:
#         if not damage_info_list:
#             return 0.0

#         # Use simplified cost calculation for speed
#         base_costs = {
#             "minor": 2000,
#             "moderate": 5000,
#             "high": 15000
#         }

#         total_cost = 0
#         for damage in damage_info_list:
#             severity = damage.get("severity", "minor")
#             area_ratio = damage.get("damage_area_ratio", 0.1)
#             base_cost = base_costs.get(severity, 2000)
#             cost = base_cost * (1 + area_ratio * 2)  # Simple multiplier
#             total_cost += cost

#         return round(total_cost, 2)

#     except Exception as e:
#         logger.error(f"Error in cost estimation: {e}")
#         return 5000.0

# # ============================
# # Routes
# # ============================
# @app.route("/")
# def home():
#     return jsonify({
#         "message": "Optimized Car Damage Detection API",
#         "status": "running",
#         "version": "2.0",
#         "endpoints": ["/health", "/predict", "/vehicle-brands"],
#         "memory_usage": f"{get_memory_usage():.1f} MB"
#     })

# @app.route("/health")
# def health():
#     """Enhanced health check with model status"""
#     health_status = {
#         "status": "healthy" if models_loaded else "degraded",
#         "models_loaded": models_loaded,
#         "service": "Car Damage Detection API",
#         "memory_usage": f"{get_memory_usage():.1f} MB",
#         "available_memory": f"{psutil.virtual_memory().available / 1024 / 1024:.1f} MB"
#     }

#     status_code = 200 if models_loaded else 503
#     return jsonify(health_status), status_code

# @app.route("/vehicle-brands")
# def get_vehicle_brands():
#     brands_models = {
#         "Toyota": ["Fortuner", "Innova", "Glanza", "Camry", "Corolla"],
#         "Hyundai": ["Creta", "Venue", "i20", "i10", "Verna"],
#         "Tata": ["Harrier", "Nexon", "Punch", "Safari", "Tiago"],
#         "Honda": ["City", "Civic", "Amaze", "Accord", "CR-V"],
#         "Kia": ["Seltos", "Sonet", "Carens", "Carnival"],
#         "Mahindra": ["XUV500", "Scorpio", "Bolero", "Thar"],
#         "Ford": ["Ecosport", "Endeavour", "Figo", "Aspire"],
#         "MarutiSuzuki": ["Swift", "Baleno", "Dzire", "WagonR"],
#     }
#     return jsonify(brands_models)

# @app.route("/predict", methods=["POST"])
# def predict():
#     """Optimized prediction endpoint with timeout and error handling"""
#     request_start = time.time()

#     try:
#         # Check if models are loaded
#         if not models_loaded:
#             if not load_models():
#                 return jsonify({
#                     "success": False,
#                     "error": "AI models not available",
#                     "message": "Service temporarily unavailable"
#                 }), 503

#         # Validate request
#         if "image" not in request.files:
#             return jsonify({"error": "No image file provided"}), 400

#         file = request.files["image"]
#         if file.filename == "":
#             return jsonify({"error": "No file selected"}), 400

#         # Save and validate image
#         img_path = save_image(file)
#         if not img_path:
#             return jsonify({"error": "Invalid file type or size. Allowed: png, jpg, jpeg (max 8MB)"}), 400

#         logger.info(f"Processing image: {img_path}, Memory: {get_memory_usage():.1f} MB")

#         # Extract vehicle details
#         user_vehicle = {
#             "brand": request.form.get("brand", "Toyota"),
#             "model": request.form.get("model", "Fortuner"),
#             "year": int(request.form.get("year", 2020)),
#             "fuel": request.form.get("fuel", "Petrol"),
#             "type": request.form.get("type", "SUV"),
#             "color": request.form.get("color", "White"),
#         }

#         # Run damage analysis with timeout
#         analysis_start = time.time()
#         damage_results = analyze_damage_optimized(img_path)

#         if not damage_results:
#             # Clean up and return no damage response
#             os.remove(img_path)
#             return jsonify({
#                 "success": True,
#                 "message": "No damage detected in the image",
#                 "damage_count": 0,
#                 "total_cost": 0,
#                 "cost_results": [],
#                 "crops": [],
#                 "annotated_image": None,
#                 "vehicle_info": user_vehicle,
#                 "currency": "INR",
#                 "processing_time": f"{time.time() - request_start:.2f}s"
#             })

#         # Create simplified annotated image
#         annotated_image = cv2.imread(img_path)
#         if annotated_image is not None:
#             annotated_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

#             # Draw bounding boxes
#             for damage in damage_results:
#                 x1, y1, x2, y2 = damage["box"]
#                 label = f"{damage['damage_type']} ({damage['severity']})"
#                 cv2.rectangle(annotated_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)
#                 cv2.putText(annotated_rgb, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

#             # Save annotated image
#             annotated_filename = f"{uuid.uuid4()}_annotated.jpg"
#             annotated_path = os.path.join(RESULTS_FOLDER, annotated_filename)
#             cv2.imwrite(annotated_path, cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR))
#             annotated_image_url = f"static/results/{annotated_filename}"
#         else:
#             annotated_image_url = None

#         # Calculate costs
#         total_cost = estimate_cost_optimized(damage_results, user_vehicle)

#         # Prepare cost results
#         cost_results = []
#         for damage in damage_results:
#             cost_results.append({
#                 "damage_type": damage["damage_type"],
#                 "severity": damage["severity"],
#                 "confidence": round(damage["confidence"], 2),
#                 "cost": round(total_cost / len(damage_results), 2),
#             })

#         # Clean up uploaded image
#         os.remove(img_path)

#         # Force garbage collection
#         gc.collect()

#         processing_time = time.time() - request_start
#         logger.info(f"Prediction completed in {processing_time:.2f}s: {len(damage_results)} damages, total cost {total_cost:.2f}")

#         response = {
#             "success": True,
#             "damage_count": len(damage_results),
#             "total_cost": round(total_cost, 2),
#             "cost_results": cost_results,
#             "crops": [],  # Simplified - no crop images to save time
#             "annotated_image": annotated_image_url,
#             "vehicle_info": user_vehicle,
#             "currency": "INR",
#             "processing_time": f"{processing_time:.2f}s",
#             "memory_usage": f"{get_memory_usage():.1f} MB"
#         }

#         return jsonify(response)

#     except Exception as e:
#         logger.error(f"Prediction error: {e}")

#         # Clean up any uploaded files on error
#         try:
#             if 'img_path' in locals() and os.path.exists(img_path):
#                 os.remove(img_path)
#         except:
#             pass

#         # Force garbage collection on error
#         gc.collect()

#         return jsonify({
#             "success": False,
#             "error": "Processing failed",
#             "message": "Unable to process the image. Please try again.",
#             "processing_time": f"{time.time() - request_start:.2f}s"
#         }), 500

# # ============================
# # Static file routes
# # ============================
# @app.route("/static/<path:path>")
# def serve_static(path):
#     return send_from_directory("static", path)

# @app.route("/static/uploads/<path:filename>")
# def serve_upload(filename):
#     return send_from_directory(UPLOAD_FOLDER, filename)

# @app.route("/static/results/<path:filename>")
# def serve_result(filename):
#     return send_from_directory(RESULTS_FOLDER, filename)

# # ============================
# # Error handlers
# # ============================
# @app.errorhandler(413)
# def too_large(e):
#     return jsonify({"error": "File too large. Maximum size is 8MB"}), 413

# @app.errorhandler(500)
# def internal_error(e):
#     return jsonify({"error": "Internal server error"}), 500

# @app.errorhandler(404)
# def not_found(e):
#     return jsonify({"error": "Endpoint not found"}), 404

# @app.errorhandler(405)
# def method_not_allowed(e):
#     return jsonify({"error": "Method not allowed"}), 405

# # ============================
# # Background cleanup
# # ============================
# def background_cleanup():
#     """Run cleanup tasks in background"""
#     while True:
#         time.sleep(1800)  # Run every 30 minutes
#         cleanup_old_files()
#         gc.collect()

# # Start background cleanup thread
# cleanup_thread = Thread(target=background_cleanup, daemon=True)
# cleanup_thread.start()

# # ============================
# # Entry point
# # ============================
# if __name__ == "__main__":
#     # Load models on startup
#     if load_models():
#         logger.info("Models loaded successfully on startup")
#     else:
#         logger.warning("Failed to load models on startup - will attempt on first request")

#     port = int(os.environ.get("PORT", 5000))
#     logger.info(f"Starting Optimized Car Damage Detection API Server on port {port}...")
#     app.run(host="0.0.0.0", port=port, debug=False, threaded=True)



# app.py
# Fixed backend - lazy model loading, CPU-only YOLO, clear logging, and safe CORS.
# Replace your existing file with this and deploy.

from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import joblib
import onnxruntime as ort
import os
import uuid
from werkzeug.utils import secure_filename
import logging
import gzip
import psutil

# ============================================
# LOGGING SETUP
# ============================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ============================================
# CORS CONFIG
# ============================================
CORS(app, resources={
    r"/*": {
        "origins": [
            "http://localhost:5173",
            "https://vd-dlproject.vercel.app",
            "https://vd-dlproject.vercel.app/",
            "https://vehicle-damage-dl-backend.onrender.com"
        ],
        "supports_credentials": True
    }
})

# ============================================
# PATHS
# ============================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "Final_Damage")

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static/uploads")
RESULTS_FOLDER = os.path.join(BASE_DIR, "static/results")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# ============================================
# GLOBALS
# ============================================
damage_session = None
severity_session = None
cost_model = None
encoders = None
feature_cols = None
models_loaded = False

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# ============================================
# YOLO ONNX HELPERS
# ============================================
def preprocess(image):
    img = cv2.resize(image, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1)
    img = img[np.newaxis, :, :, :] / 255.0
    return img.astype(np.float32)

def xywh2xyxy(x):
    y = np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y

def nms(boxes, scores, iou_threshold=0.5):
    idxs = scores.argsort()[::-1]
    keep = []

    while idxs.size > 0:
        i = idxs[0]
        keep.append(i)

        xx1 = np.maximum(boxes[i][0], boxes[idxs[1:]][:, 0])
        yy1 = np.maximum(boxes[i][1], boxes[idxs[1:]][:, 1])
        xx2 = np.minimum(boxes[i][2], boxes[idxs[1:]][:, 2])
        yy2 = np.minimum(boxes[i][3], boxes[idxs[1:]][:, 3])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h

        union = (
            (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1])
            + (boxes[idxs[1:]][:, 2] - boxes[idxs[1:]][:, 0]) * (boxes[idxs[1:]][:, 3] - boxes[idxs[1:]][:, 1])
            - inter
        )

        iou = inter / (union + 1e-6)
        idxs = idxs[1:][iou < iou_threshold]

    return keep

def postprocess(outputs, conf=0.25):
    predictions = outputs[0].squeeze(0)

    boxes = predictions[:, :4]
    scores = np.max(predictions[:, 4:], axis=1)
    classes = np.argmax(predictions[:, 4:], axis=1)

    mask = scores > conf
    boxes, scores, classes = boxes[mask], scores[mask], classes[mask]

    boxes = xywh2xyxy(boxes)
    keep = nms(boxes, scores)

    return [{
        "damage_type": str(classes[i]),
        "severity": "low" if scores[i] < 0.75 else "medium" if scores[i] < 0.90 else "high",
        "confidence": float(scores[i]),
        "box": [int(v) for v in boxes[i]]
    } for i in keep]

# ============================================
# MODEL LOADING
# ============================================
def load_models():
    global damage_session, severity_session, cost_model, encoders, feature_cols, models_loaded

    try:
        logger.info("========== MODEL LOADING STARTED ==========")

        damage_session = ort.InferenceSession(os.path.join(MODEL_DIR, "DamageTypebest.onnx"))
        severity_session = ort.InferenceSession(os.path.join(MODEL_DIR, "Severitybest.onnx"))

        with gzip.open(os.path.join(MODEL_DIR, "cost_model.pkl.gz"), "rb") as f:
            cost_model = joblib.load(f)

        encoders = joblib.load(os.path.join(MODEL_DIR, "label_encoders.pkl"))
        feature_cols = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))

        models_loaded = True
        logger.info("🚀 ALL MODELS LOADED SUCCESSFULLY")
        return True

    except Exception as e:
        logger.error(f"🔥 MODEL LOADING FAILED: {str(e)}")
        models_loaded = False
        return False

# ============================================
# UTILS
# ============================================
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def save_image(file):
    filename = secure_filename(file.filename)
    path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}_{filename}")
    file.save(path)
    return path

# ============================================
# ROUTES
# ============================================
@app.route("/")
def home():
    return jsonify({"service": "Vehicle Damage API", "status": "running", "models_loaded": models_loaded})

@app.route("/health")
def health():
    return jsonify({"status": "healthy"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if not models_loaded:
            load_models()

        if "image" not in request.files:
            return jsonify({"error": "No image provided"}), 400

        path = save_image(request.files["image"])
        damages = postprocess(damage_session.run(None, {damage_session.get_inputs()[0].name: preprocess(cv2.imread(path))}))

        return jsonify({
            "success": True,
            "damage_count": len(damages),
            "damages": damages,
            "total_cost": sum({"low":2000, "medium":5000, "high":15000}[d["severity"]] for d in damages),
            "currency": "INR"
        })

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": "Prediction failed"}), 500

# ============================================
# FORCE CORS HEADERS
# ============================================
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "https://vd-dlproject.vercel.app"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    return response

# ============================================
# START SERVER
# ============================================
if __name__ == "__main__":
    load_models()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)

