import os
import logging
from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
from processing.predictor import predict_forgery

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def home():
    return render_template("upload.html")

@app.route("/upload", methods=["POST"])
def upload():
    try:
        if "file" not in request.files:
            logger.warning("No file part in request")
            return "No file uploaded", 400
            
        file = request.files.get("file")

        if not file or file.filename == "":
            logger.warning("Empty filename uploaded")
            return "No file selected", 400

        if not allowed_file(file.filename):
            logger.warning(f"Unsupported file type: {file.filename}")
            return "Unsupported file type", 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        
        logger.info(f"Saving uploaded file to {filepath}")
        file.save(filepath)

        # Call prediction with error handling
        try:
            logger.info("Calling predict_forgery...")
            predicted_class, confidence, accuracy, result_image = predict_forgery(filepath)
            logger.info(f"Prediction result: {predicted_class}, Accuracy: {accuracy}")
        except Exception as e:
            logger.error(f"Critical error during prediction: {e}", exc_info=True)
            return render_template("upload.html", error="An error occurred during image analysis. Please try again.")

        # Ensure values are safe for display
        try:
            display_conf = round(float(confidence or 0) * 100, 2)
            display_acc = round(float(accuracy or 0) * 100, 2)
        except (TypeError, ValueError):
            display_conf = 0.0
            display_acc = 0.0

        return render_template(
            "result.html",
            predicted_class=predicted_class,
            confidence=display_conf,
            accuracy=display_acc,
            image=result_image,
            uploaded_image=filename,
            confusion_matrix="confusion_matrix.png"
        )
    except Exception as e:
        logger.error(f"General upload error: {e}", exc_info=True)
        return "An unexpected error occurred", 500

if __name__ == "__main__":
    # Use environment port for Render compatibility
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)