import os

import numpy as np

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash
)

from werkzeug.utils import secure_filename

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


# --------------------------------------------------
# Flask configuration
# --------------------------------------------------

app = Flask(__name__)

app.secret_key = "rice-leaf-disease-secret-key"


# --------------------------------------------------
# Upload configuration
# --------------------------------------------------

UPLOAD_FOLDER = "static/uploads"

ALLOWED_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg",
    "webp",
    "jfif"
}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# --------------------------------------------------
# Load trained model
# --------------------------------------------------

MODEL_PATH = "model/rice_leaf_mobilenetv2.keras"

model = load_model(MODEL_PATH)


# --------------------------------------------------
# Disease classes
# --------------------------------------------------

CLASS_NAMES = [
    "Bacterial Leaf Blight",
    "Brown Spot",
    "Leaf Smut"
]


# --------------------------------------------------
# Check file extension
# --------------------------------------------------

def allowed_file(filename):

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


# --------------------------------------------------
# Home page
# --------------------------------------------------

@app.route("/")
def home():

    return render_template("home.html")


# --------------------------------------------------
# Prediction
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    # Check whether file exists
    if "file" not in request.files:

        flash("Please select an image.")

        return redirect(url_for("home"))


    file = request.files["file"]


    # Check empty file
    if file.filename == "":

        flash("Please select an image.")

        return redirect(url_for("home"))


    # Check extension
    if not allowed_file(file.filename):

        flash(
            "Invalid file format. "
            "Please upload JPG, JPEG, PNG, WEBP or JFIF"
        )

        return redirect(url_for("home"))


    # Secure filename
    filename = secure_filename(file.filename)


    # Save uploaded image
    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    file.save(filepath)


    # --------------------------------------------------
    # Load image
    # --------------------------------------------------

    img = image.load_img(
        filepath,
        target_size=(224, 224)
    )


    # Convert image to numpy array
    img_array = image.img_to_array(img)


    # Add batch dimension
    img_array = np.expand_dims(
        img_array,
        axis=0
    )


    # --------------------------------------------------
    # IMPORTANT
    # Same preprocessing as training:
    # rescale=1./255
    # --------------------------------------------------

    img_array = img_array / 255.0


    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------

    predictions = model.predict(img_array)


    # Get predicted class
    predicted_index = np.argmax(predictions[0])


    predicted_class = CLASS_NAMES[predicted_index]


    # Confidence
    confidence = float(
        predictions[0][predicted_index] * 100
    )


    # --------------------------------------------------
    # Result
    # --------------------------------------------------

    return render_template(
        "results.html",
        prediction=predicted_class,
        confidence=round(confidence, 2),
        image_path=filename
    )


# --------------------------------------------------
# Run Flask
# --------------------------------------------------

if __name__ == "__main__":

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    ) 