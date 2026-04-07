import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from processing.predictor import predict_forgery

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
    file = request.files.get("file")

    if not file or file.filename == "":
        return "No file uploaded", 400

    if not allowed_file(file.filename):
        return "Unsupported file type", 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    predicted_class, confidence, accuracy, image_name = predict_forgery(filepath)

    return render_template(
        "result.html",
        predicted_class=predicted_class,
        confidence=round(confidence * 100, 2),
        accuracy=round(accuracy * 100, 2),
        image=image_name,
        uploaded_image=filename,
        confusion_matrix="confusion_matrix.png"
    )


if __name__ == "__main__":
    app.run(debug=True)