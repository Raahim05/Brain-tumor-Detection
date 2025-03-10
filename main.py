from fastapi import FastAPI, Request, UploadFile, File
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import model_from_json, Model
import tensorflow as tf
import numpy as np
import cv2
from io import BytesIO
from PIL import Image
import os
import uuid
import time
import shutil
from pathlib import Path

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Directory setup
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
STATIC_DIR = BASE_DIR / "static"
TEMP_DIR = BASE_DIR / "temp"  # New temp directory for processing

for directory in [UPLOAD_DIR, STATIC_DIR, TEMP_DIR]:
    directory.mkdir(exist_ok=True)

# Load templates
templates = Jinja2Templates(directory="templates")

# Load the model (unchanged)
json_file = open("model.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")

# Existing TensorFlow functions (unchanged)
@tf.function(reduce_retracing=True)
def predict_tumor(image_tensor: tf.Tensor) -> tf.Tensor:
    return loaded_model(image_tensor)

@tf.function(reduce_retracing=True)
def predict_localization(model: Model, image_tensor: tf.Tensor) -> tf.Tensor:
    return model(image_tensor)

# Existing preprocessing and validation functions (unchanged)
def preprocess_image(image: Image.Image) -> tf.Tensor:
    image = image.convert("RGB")
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
    return tf.expand_dims(image_tensor, axis=0)

def is_mri_image(image: Image.Image):
    img_array = np.array(image)
    if len(img_array.shape) == 2:
        return True
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        std_dev = np.std(gray)
        if std_dev < 40 or (np.allclose(img_array[:, :, 0], img_array[:, :, 1], atol=10) and 
                            np.allclose(img_array[:, :, 1], img_array[:, :, 2], atol=10)):
            return True
    if np.std(img_array) < 20:
        return True
    return False

def get_localization_layer(model):
    for layer in model.layers[::-1]:
        if 'conv' in layer.name.lower():
            return Model(inputs=model.input, outputs=layer.output)
    return None

def mark_affected_region(image_path, output_path):
    # (Unchanged - keeping your existing implementation)
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image at {image_path}")
    
    img_resized = cv2.resize(img, (224, 224))
    img_normalized = img_resized / 255.0
    img_input = tf.convert_to_tensor(np.expand_dims(img_normalized, axis=0), dtype=tf.float32)

    localization_model = get_localization_layer(loaded_model)
    if localization_model is None:
        height, width, _ = img.shape
        cv2.rectangle(img, (50, 50), (width - 50, height - 50), (0, 0, 255), 3)
        cv2.putText(img, "Tumor Region (Approx)", (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imwrite(output_path, img)
        return

    activation_map = predict_localization(localization_model, img_input)[0]
    activation_map = np.mean(activation_map, axis=-1)
    activation_map_resized = cv2.resize(activation_map, (224, 224))
    activation_map_normalized = cv2.normalize(activation_map_resized, None, 0, 1, cv2.NORM_MINMAX)
    heatmap = cv2.applyColorMap(np.uint8(255 * activation_map_normalized), cv2.COLORMAP_JET)
    original_resized = cv2.resize(img, (224, 224))
    overlay = cv2.addWeighted(original_resized, 0.7, heatmap, 0.3, 0)
    threshold = np.max(activation_map_normalized) * 0.75
    binary_map = (activation_map_normalized > threshold).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    binary_map = cv2.morphologyEx(binary_map, cv2.MORPH_OPEN, kernel)
    binary_map = cv2.morphologyEx(binary_map, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        mask = np.zeros_like(original_resized)
        cv2.drawContours(mask, contours, -1, (0, 255, 0), 2)
        largest_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(overlay, [box], 0, (0, 255, 0), 2)
        x, y, w, h = cv2.boundingRect(largest_contour)
        cv2.putText(overlay, "Tumor Region", (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        final_image = cv2.resize(overlay, (img.shape[1], img.shape[0]))
    else:
        height, width = img.shape[:2]
        cv2.rectangle(img, (width//3, height//3), (2*width//3, 2*height//3), 
                     (0, 255, 0), 2)
        cv2.putText(img, "Approximate Tumor Region", (width//3, height//3-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        final_image = img
    cv2.imwrite(output_path, final_image)

# New file management function
def cleanup_temp_files(directory: Path, max_age: int = 3600):
    """Remove files older than max_age seconds from the specified directory."""
    current_time = time.time()
    for file_path in directory.iterdir():
        if file_path.is_file():
            file_age = current_time - file_path.stat().st_mtime
            if file_age > max_age:
                try:
                    file_path.unlink()
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the home page for file upload."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, files: list[UploadFile] = File(...)):
    """Predict the presence of a brain tumor and mark affected regions."""
    try:
        predictions = []
        session_id = str(uuid.uuid4())  # Unique session ID for this request
        session_temp_dir = TEMP_DIR / session_id
        session_temp_dir.mkdir(exist_ok=True)

        # Clean up old temp files (runs on each request, could be scheduled separately)
        cleanup_temp_files(TEMP_DIR)

        for file in files:
            # Generate unique filename
            original_ext = os.path.splitext(file.filename)[1]
            unique_filename = f"{uuid.uuid4()}_{int(time.time())}{original_ext}"
            temp_path = session_temp_dir / unique_filename
            static_path = STATIC_DIR / unique_filename

            # Save to temp directory
            with open(temp_path, "wb") as f:
                f.write(await file.read())

            # Move to static for serving
            shutil.move(temp_path, static_path)

            if not static_path.exists():
                raise FileNotFoundError(f"File {unique_filename} not found in static")

            try:
                image = Image.open(static_path)
            except Exception as e:
                print(f"Error opening image {static_path}: {str(e)}")
                raise

            if not is_mri_image(image):
                predictions.append({
                    "error": "Uploaded image is not an MRI scan!",
                    "image_path": f"/static/{unique_filename}"
                })
                continue

            processed_image = preprocess_image(image)
            prediction = predict_tumor(processed_image)
            tumor_prob = float(prediction[0][1])
            result = "Tumor Detected" if tumor_prob > 0.5 else "No Tumor Detected"
            confidence = round(tumor_prob * 100, 2) if result == "Tumor Detected" else round((1 - tumor_prob) * 100, 2)

            marked_path = None
            if result == "Tumor Detected":
                marked_filename = f"marked_{unique_filename}"
                marked_path = STATIC_DIR / marked_filename
                try:
                    mark_affected_region(str(static_path), str(marked_path))
                except Exception as e:
                    print(f"Error marking region for {unique_filename}: {str(e)}")
                    raise
                marked_path = f"/static/{marked_filename}"

            predictions.append({
                "result": result,
                "confidence": confidence,
                "original_path": f"/static/{unique_filename}",
                "marked_path": marked_path,
                "tumor_present": result == "Tumor Detected",
                "session_id": session_id  # Pass session ID for potential cleanup
            })

        mri_predictions = [img for img in predictions if "error" not in img]
        tumor_count = sum(1 for img in mri_predictions if img["tumor_present"])
        total_mri_images = len(mri_predictions)
        avg_confidence = round(sum(img["confidence"] for img in mri_predictions) / total_mri_images, 2) if total_mri_images > 0 else 0

        return templates.TemplateResponse(
            "result.html", 
            {
                "request": request,
                "images": predictions,
                "tumor_count": tumor_count,
                "total_mri_images": total_mri_images,
                "avg_confidence": avg_confidence,
                "session_id": session_id  # Pass to template if needed
            }
        )
    except Exception as e:
        print(f"Error in predict: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Optional: Cleanup endpoint (for manual or scheduled cleanup)
@app.get("/cleanup/{session_id}")
async def cleanup_session(session_id: str):
    """Remove files associated with a specific session."""
    session_dir = TEMP_DIR / session_id
    if session_dir.exists():
        shutil.rmtree(session_dir)
    # Clean up static files linked to this session (if tracked)
    return {"message": f"Cleaned up session {session_id}"}