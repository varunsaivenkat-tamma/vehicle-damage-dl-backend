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

import os
import gc
import time
import gzip
import logging
import uuid

from threading import Thread
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# IMPORTANT: set this before importing/using models to avoid CUDA attempts
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # force CPU (empty = no GPU)

# Import heavy libs after env tweak
import cv2
import joblib
import psutil
from ultralytics import YOLO  # ultralytics may still allocate memory on load -> handled carefully

# -----------------------
# Logging and app setup
# -----------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vehicle-damage-api")

app = Flask(__name__)

# -----------------------
# CORS - explicit domains (no wildcard with supports_credentials)
# -----------------------
CORS(app, resources={
    r"/*": {
        "origins": [
            "http://localhost:5173",
            "https://vehicle-damage-dl-frontend.onrender.com",
            "https://vd-dlproject.vercel.app",
            "https://vehicle-damage-dl-backend.onrender.com"
        ],
        "supports_credentials": True,
        "allow_headers": ["Content-Type", "Authorization"],
        "methods": ["GET", "POST", "OPTIONS"]
    }
})

# -----------------------
# Paths and folders
# -----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "Final_Damage")  # must match your folder name exactly
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
RESULTS_FOLDER = os.path.join(BASE_DIR, "static", "results")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# -----------------------
# Globals
# -----------------------
damage_model = None
severity_model = None
cost_model = None
encoders = None
feature_cols = None
models_loaded = False

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB upload limit

# -----------------------
# Helpers
# -----------------------
def get_memory_usage_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def save_image(file):
    if not allowed_file(file.filename):
        return None
    filename = secure_filename(file.filename)
    unique = f"{uuid.uuid4()}_{filename}"
    path = os.path.join(UPLOAD_FOLDER, unique)
    file.save(path)
    return path

def cleanup_old_files(max_age_seconds=3600):
    """Delete files older than max_age_seconds in uploads/results to save disk."""
    now = time.time()
    for folder in (UPLOAD_FOLDER, RESULTS_FOLDER):
        try:
            for fname in os.listdir(folder):
                fp = os.path.join(folder, fname)
                if os.path.isfile(fp) and now - os.path.getmtime(fp) > max_age_seconds:
                    os.remove(fp)
                    logger.info(f"Removed old file: {fp}")
        except Exception:
            pass

# -----------------------
# Model loading (lazy, CPU-only)
# -----------------------
def load_models():
    """
    Try to load all models. Returns True on success, False on failure.
    This function logs the precise exception so you can see memory / missing-file errors.
    """
    global damage_model, severity_model, cost_model, encoders, feature_cols, models_loaded

    if models_loaded:
        logger.info("Models already loaded - skipping.")
        return True

    logger.info("Attempting to load models...")
    gc.collect()

    # Print model dir + contents for debugging
    logger.info(f"MODEL_DIR resolved to: {MODEL_DIR}")
    if not os.path.exists(MODEL_DIR):
        logger.error("MODEL_DIR does not exist. Check name/path.")
        models_loaded = False
        return False

    try:
        files = os.listdir(MODEL_DIR)
        logger.info(f"Files in MODEL_DIR: {files}")
    except Exception as e:
        logger.error(f"Could not list MODEL_DIR: {e}")
        models_loaded = False
        return False

    # Compose paths
    damage_path = os.path.join(MODEL_DIR, "DamageTypebest.pt")
    severity_path = os.path.join(MODEL_DIR, "Severitybest.pt")
    cost_path = os.path.join(MODEL_DIR, "cost_model.pkl.gz")
    encoders_path = os.path.join(MODEL_DIR, "label_encoders.pkl")
    feature_cols_path = os.path.join(MODEL_DIR, "feature_columns.pkl")

    # Validate files exist before trying to load
    for p in (damage_path, severity_path, cost_path, encoders_path, feature_cols_path):
        if not os.path.exists(p):
            logger.error(f"Required model file missing: {p}")
            models_loaded = False
            return False

    try:
        # Force CPU usage by re-setting env and letting ultralytics use CPU
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        logger.info("Creating YOLO(damage) model (CPU mode)...")
        damage_model = YOLO(damage_path)  # keep this inside try/except
        logger.info("Creating YOLO(severity) model (CPU mode)...")
        severity_model = YOLO(severity_path)

        logger.info("Loading compressed cost model...")
        with gzip.open(cost_path, "rb") as f:
            cost_model = joblib.load(f)

        logger.info("Loading encoders and feature columns...")
        encoders = joblib.load(encoders_path)
        feature_cols = joblib.load(feature_cols_path)

        models_loaded = True
        logger.info("Models loaded successfully.")
        return True

    except MemoryError as me:
        logger.error("MemoryError while loading models. Likely OOM on the host (not enough RAM).")
        logger.error(str(me))
        models_loaded = False
        return False

    except Exception as e:
        # Catch any other exceptions (missing file, format, model incompatibility)
        logger.error(f"Exception during model loading: {e}", exc_info=True)
        models_loaded = False
        return False

# -----------------------
# Simple analysis & cost (small, fast)
# -----------------------
def analyze_damage_fast(img_path):
    """
    Very simple/fast analysis to avoid heavy compute on the server.
    Uses the loaded damage_model (YOLO) if available; otherwise returns empty.
    """
    global damage_model
    try:
        if damage_model is None:
            logger.warning("damage_model is None - cannot analyze.")
            return []

        img = cv2.imread(img_path)
        if img is None:
            logger.error("Could not read uploaded image for analysis.")
            return []

        preds = damage_model(img, conf=0.25, verbose=False)
        if len(preds) == 0 or len(preds[0].boxes) == 0:
            return []

        results = []
        # limit to top 3 detections to save CPU/memory/time
        for box in preds[0].boxes[:3]:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = damage_model.names[cls] if hasattr(damage_model, "names") else str(cls)
            sev = "minor"
            if conf > 0.75:
                sev = "moderate"
            if conf > 0.85:
                sev = "high"
            results.append({
                "damage_type": label,
                "severity": sev,
                "confidence": round(conf, 2),
                "box": [int(x1), int(y1), int(x2), int(y2)]
            })
        return results

    except Exception as e:
        logger.error(f"Error in analyze_damage_fast: {e}", exc_info=True)
        return []

def estimate_cost_simple(damages, vehicle_info):
    if not damages:
        return 0
    base = {"minor": 2000, "moderate": 5000, "high": 15000}
    total = sum(base.get(d["severity"], 2000) for d in damages)
    return round(total, 2)

# -----------------------
# Routes
# -----------------------
@app.route("/")
def home():
    return jsonify({
        "service": "Vehicle Damage Detection API",
        "status": "running",
        "models_loaded": models_loaded,
        "memory_mb": f"{get_memory_usage_mb():.1f}"
    })

@app.route("/health")
def health():
    """
    Returns 200 when models_loaded==True, otherwise 503.
    Frontend should treat models_loaded:False as 'degraded/starting' but the server is reachable.
    """
    return jsonify({
        "status": "healthy" if models_loaded else "degraded",
        "models_loaded": models_loaded,
        "memory_mb": f"{get_memory_usage_mb():.1f}"
    }), 200 if models_loaded else 503

@app.route("/load-models", methods=["POST", "GET"])
def load_models_endpoint():
    """
    Manual trigger for loading models (useful when you want to load on-demand).
    Example: POST https://your-backend/load-models
    """
    ok = load_models()
    if ok:
        return jsonify({"success": True, "message": "Models loaded"}), 200
    else:
        return jsonify({"success": False, "message": "Model loading failed - check logs"}), 500

@app.route("/predict", methods=["POST"])
def predict():
    """
    On first request this will attempt to load models. If the host cannot load (OOM), it
    will return a friendly error. If models load successfully, it will perform a quick
    detection and return cost estimate.
    """
    start = time.time()
    # Lazy load: try to load models if not already loaded
    if not models_loaded:
        logger.info("models_loaded=False -> attempting to load models inside /predict")
        if not load_models():
            # If loading fails, return helpful message and logs advice
            logger.error("Model loading failed inside /predict. Returning 503.")
            return jsonify({
                "success": False,
                "error": "Models not available",
                "message": "Server could not load AI models (likely insufficient memory or missing files). Check backend logs."
            }), 503

    # Validate image
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"success": False, "error": "Empty filename"}), 400

    path = save_image(file)
    if not path:
        return jsonify({"success": False, "error": "Invalid file type"}), 400

    # Run a quick analysis (limited detections)
    results = analyze_damage_fast(path)

    # Prepare vehicle info
    vehicle = {
        "brand": request.form.get("brand", "Toyota"),
        "model": request.form.get("model", "Fortuner"),
        "year": int(request.form.get("year", 2020)),
        "fuel": request.form.get("fuel", "Petrol"),
        "type": request.form.get("type", "SUV"),
        "color": request.form.get("color", "White"),
    }

    total_cost = estimate_cost_simple(results, vehicle)

    # Clean-up uploaded file after processing
    try:
        os.remove(path)
    except Exception:
        pass

    elapsed = time.time() - start
    logger.info(f"Predict completed in {elapsed:.2f}s - detections: {len(results)} - memory: {get_memory_usage_mb():.1f}MB")

    return jsonify({
        "success": True,
        "damage_count": len(results),
        "damages": results,
        "total_cost": total_cost,
        "vehicle_info": vehicle,
        "currency": "INR",
        "processing_time_s": round(elapsed, 2)
    }), 200

# Static serving (optional)
@app.route("/static/<path:path>")
def serve_static(path):
    return send_from_directory(os.path.join(BASE_DIR, "static"), path)

# -----------------------
# Background cleanup thread (keeps disk tidy)
# -----------------------
def periodic_cleanup():
    while True:
        try:
            cleanup_old_files(max_age_seconds=3600)
            gc.collect()
        except Exception:
            pass
        time.sleep(1800)  # 30 minutes

cleanup_thread = Thread(target=periodic_cleanup, daemon=True)
cleanup_thread.start()

# -----------------------
# Entry point - do NOT load models here on start (avoid OOM)
# -----------------------
if __name__ == "__main__":
    logger.info("Starting server (models WILL be loaded lazily on first request or via /load-models).")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
