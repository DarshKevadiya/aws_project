
from io import BytesIO
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array, load_img

app = Flask(__name__)
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = next(
    (
        path
        for path in (
            PROJECT_DIR / "model" / "garbage_model.h5",
            PROJECT_DIR / "model" / "garbage_model.keras",
        )
        if path.is_file()
    ),
    PROJECT_DIR / "model" / "garbage_model.h5",
)
model = load_model(MODEL_PATH)
input_height, input_width = model.input_shape[1:3]

dataset_candidates = (
    PROJECT_DIR / "dataset",
    PROJECT_DIR / "Garbage classification" / "Garbage classification",
)
for dataset_dir in dataset_candidates:
    if not dataset_dir.is_dir():
        continue
    class_dir = (
        dataset_dir / "train"
        if (dataset_dir / "train").is_dir()
        else dataset_dir
    )
    CLASS_NAMES = sorted(
        path.name
        for path in class_dir.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )
    if CLASS_NAMES:
        break
else:
    raise FileNotFoundError("Could not find the model's class folders.")

if len(CLASS_NAMES) != model.output_shape[-1]:
    raise ValueError("The class folders do not match the model output size.")


@app.route("/")
def index():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    uploaded_file = request.files.get("image")
    if uploaded_file is None or not uploaded_file.filename:
        return jsonify({"error": "Upload an image using the 'image' field."}), 400

    try:
        image = load_img(
            BytesIO(uploaded_file.read()),
            target_size=(input_height, input_width),
            color_mode="rgb",
        )
    except (OSError, ValueError):
        return jsonify({"error": "The uploaded file is not a valid image."}), 400

    image_array = img_to_array(image)
    image_array = preprocess_input(image_array)
    probabilities = model.predict(image_array[None, ...], verbose=0)[0]
    predicted_index = int(probabilities.argmax())

    return jsonify(
        {
            "class_name": CLASS_NAMES[predicted_index],
            "confidence": float(probabilities[predicted_index]),
        }
    )


# from flask import Flask

# # Initialize the Flask application
# app = Flask(__name__)

# # Define the route for the home page
# @app.route('/')
# def hello_world():
#     return 'Hello, World!'

# Run the local development server
# if __name__ == '__main__':
#     app.run(debug=True)
