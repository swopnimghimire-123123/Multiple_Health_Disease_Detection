import os
import uuid
import pickle
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename
import tensorflow as tf
import warnings


warnings.filterwarnings('ignore')

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
# Model was trained on 150x150 RGB images with label order defined in the notebook.
BRAIN_CLASSES = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]
IMAGE_SIZE = 150
BRAIN_CLASS_DISPLAY = {
    "glioma_tumor": "Tumor: Glioma",
    "meningioma_tumor": "Tumor: Meningioma",
    "no_tumor": "No Tumor",
    "pituitary_tumor": "Tumor: Pituitary",
}
brain_input_size = (IMAGE_SIZE, IMAGE_SIZE)
brain_input_channels = 3

app = Flask(__name__, template_folder="templetes", static_folder="static")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
app.secret_key = "demo-safe-secret"

brain_model = None
heart_model = None
diabetes_model = None

print("\n" + "=" * 80)
print("FLASK APP STARTUP - LOADING MODELS")
print("=" * 80)

# Load Brain Model
print("\n[1/3] Loading Brain Tumor Model...")

try:
    brain_model_path = os.path.join(BASE_DIR, "models", "braintumor.h5")
    brain_model = tf.keras.models.load_model(brain_model_path)
    # If the model encodes an explicit spatial input size, adopt it to avoid shape mismatches.
    if hasattr(brain_model, "input_shape") and brain_model.input_shape and len(brain_model.input_shape) >= 4:
        _, h, w, c = brain_model.input_shape
        if h and w:
            brain_input_size = (int(h), int(w))
        if c:
            brain_input_channels = int(c)
        print(f"Using brain model input size: {brain_input_size} and channels: {brain_input_channels}")
    print("✅ BRAIN MODEL LOADED")
except Exception as exc:
    print(f"❌ BRAIN MODEL FAILED: {exc}")
    brain_model = None
    
    
# Load Heart Model
print("\n[2/3] Loading Heart Disease Model...")
try:
    heart_model_path = os.path.join(BASE_DIR, "models", "heart_disease_model.sav")
    with open(heart_model_path, "rb") as f:
        heart_model = pickle.load(f)
    print("✅ HEART MODEL LOADED")
except Exception as exc:
    print(f"❌ HEART MODEL FAILED: {exc}")
    heart_model = None

# Load Diabetes Model
print("\n[3/3] Loading Diabetes Model...")
try:
    diabetes_model_path = os.path.join(BASE_DIR, "models", "diabetes.pkl")
    with open(diabetes_model_path, "rb") as f:
        diabetes_model = pickle.load(f)
    print("✅ DIABETES MODEL LOADED")
except Exception as exc:
    print(f"❌ DIABETES MODEL FAILED: {exc}")
    diabetes_model = None

print("\n" + "=" * 80)
print("MODEL LOADING COMPLETE")
print(f"Brain Model: {'✅ LOADED' if brain_model else '❌ FAILED'}")
print(f"Heart Model: {'✅ LOADED' if heart_model else '❌ FAILED'}")
print(f"Diabetes Model: {'✅ LOADED' if diabetes_model else '❌ FAILED'}")
print("=" * 80)

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_brain_image(path: str) -> np.ndarray:
    """Preprocess image for brain tumor model"""
    image = Image.open(path).convert("RGB")
    image = image.resize(brain_input_size)

    if brain_input_channels == 1:
        # Convert to grayscale if the model expects a single channel.
        image = image.convert("L")
        arr = np.asarray(image, dtype=np.float32)
        arr = np.expand_dims(arr, axis=-1)
    else:
        arr = np.asarray(image, dtype=np.float32)
        # Model was trained using cv2 (BGR) without normalization; mirror that here.
        arr = arr[..., ::-1]

    return np.expand_dims(arr, axis=0)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/model-info")
def model_info():
    return render_template("model_info.html")


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/brain-tumor", methods=["GET", "POST"])
def brain_tumor():
    prediction_text = None
    confidence = None
    image_url = None

    if request.method == "POST":
        file = request.files.get("image")
        if not file or file.filename == "":
            flash("Please choose an image file.", "error")
            return redirect(request.url)

        if not allowed_file(file.filename):
            flash("Only JPG and PNG files are allowed.", "error")
            return redirect(request.url)

        if brain_model is None:
            flash("Model not available. Please check server logs.", "error")
            return redirect(request.url)

        try:
            filename = secure_filename(file.filename)
            unique_name = f"{uuid.uuid4().hex}_{filename}"
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
            file.save(save_path)

            # Preprocess image
            img_array = preprocess_brain_image(save_path)
            
            # Predict
            preds = brain_model.predict(img_array, verbose=0)[0]
            best_idx = int(np.argmax(preds))
            best_conf = float(preds[best_idx])
            predicted_class = BRAIN_CLASSES[best_idx]

            # Format result
            prediction_text = BRAIN_CLASS_DISPLAY.get(predicted_class, predicted_class)
            confidence = round(best_conf * 100, 2)
            image_url = url_for("uploaded_file", filename=unique_name)
            
            print(f"✅ Prediction: {prediction_text}, Confidence: {confidence}%")
            
        except Exception as exc:
            flash(f"Could not process image: {str(exc)}", "error")
            print(f"❌ Prediction error: {exc}")
            import traceback
            traceback.print_exc()
            return redirect(request.url)

    return render_template(
        "brain_tumor.html",
        prediction_text=prediction_text,
        confidence=confidence,
        image_url=image_url,
    )


@app.route("/heart-disease", methods=["GET", "POST"])
def heart_disease():
    prediction_text = None
    probability = None

    if request.method == "POST":
        fields = [
            "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal",
        ]
        try:
            values = [float(request.form.get(f, "").strip()) for f in fields]
            sample = np.array([values], dtype=np.float32)

            if heart_model is None:
                raise RuntimeError("Model not available.")

            if hasattr(heart_model, "predict_proba"):
                proba = heart_model.predict_proba(sample)[0][1]
                probability = round(float(proba) * 100, 2)
                risk = proba >= 0.5
            else:
                pred = heart_model.predict(sample)[0]
                risk = bool(pred)
                probability = None

            prediction_text = "High Risk" if risk else "Low Risk"
        except Exception as exc:
            flash(f"Unable to predict: {str(exc)}", "error")

    return render_template(
        "heart_disease.html",
        prediction_text=prediction_text,
        probability=probability,
    )


@app.route("/diabetes", methods=["GET", "POST"])
def diabetes():
    prediction_text = None
    confidence = None

    if request.method == "POST":
        fields = [
            "pregnancies", "glucose", "bloodpressure", "skinthickness",
            "insulin", "bmi", "dpf", "age",
        ]
        try:
            values = [float(request.form.get(f, "").strip()) for f in fields]
            sample = np.array([values], dtype=np.float32)

            if diabetes_model is None:
                raise RuntimeError("Model not available.")

            if hasattr(diabetes_model, "predict_proba"):
                proba = diabetes_model.predict_proba(sample)[0][1]
                confidence = round(float(proba) * 100, 2)
                diabetic = proba >= 0.5
            else:
                pred = diabetes_model.predict(sample)[0]
                diabetic = bool(pred)
                confidence = None

            prediction_text = "Diabetic" if diabetic else "Non-Diabetic"
        except Exception as exc:
            flash(f"Unable to predict: {str(exc)}", "error")

    return render_template(
        "diabetes.html",
        prediction_text=prediction_text,
        confidence=confidence,
    )


if __name__ == "__main__":
    print("\n🚀 Starting Flask server...")
    print("📱 Open http://localhost:5000 in your browser")
    print("🔧 Debug mode: ON")
    app.run(host="0.0.0.0", port=5000, debug=True)