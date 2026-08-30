import os

import cv2
import numpy as np
from flask import Flask, jsonify, render_template_string, request
from tensorflow.keras.models import load_model


MODEL_PATH = "saved_models/mini_resnet.keras"
MIN_CONFIDENCE = 80.0


data_map = {
    1: "ꠀ (aa/আ)",
    2: "ꠁ (ei/ই)",
    3: "ꠃ (u/উ)",
    4: "ꠄ (a/এ)",
    5: "ꠅ (oa/ও)",
    6: "ꠇ (ko/ক)",
    7: "ꠈ (kho/খ)",
    8: "ꠉ (go/গ)",
    9: "ꠊ (gho/ঘ)",
    10: "ꠌ (co/চ)",
    11: "ꠍ (cho/ছ)",
    12: "ꠎ (jo/জ)",
    13: "ꠏ (jho/ঝ)",
    14: "ꠐ (To/ট)",
    15: "ꠑ (Tho/ঠ)",
    16: "ꠒ (Do/ড)",
    17: "ꠓ (Dho/ঢ)",
    18: "ꠔ (to/ত)",
    19: "ꠕ (tho/থ)",
    20: "ꠖ (do/দ)",
    21: "ꠗ (dho/ধ)",
    22: "ꠘ (no/ন)",
    23: "ꠙ (po/প)",
    24: "ꠚ (pho/ফ)",
    25: "ꠛ (bo/ব)",
    26: "ꠜ (bho/ভ)",
    27: "ꠝ (mo/ম)",
    28: "ꠞ (ro/র)",
    29: "ꠟ (lo/ল)",
    30: "ꠠ (oR/ড়)",
    31: "ꠡ (sho/শ)",
    32: "ꠢ (ho/হ)"
}


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

model = None


HTML = """
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">

    <meta
        name="viewport"
        content="width=device-width, initial-scale=1.0"
    >

    <title>Sylheti Nagri Recognizer</title>

    <style>
        * {
            box-sizing: border-box;
        }

        html {
            scroll-behavior: smooth;
        }

        body {
            margin: 0;
            min-height: 100vh;
            padding: 35px 20px;
            color: #1f2937;
            background: #ffffff;
            font-family:
                Arial,
                "Noto Sans Bengali",
                sans-serif;
        }

        .container {
            width: min(850px, 100%);
            margin: auto;
        }

        h1 {
            margin: 0 0 28px;
            color: #111827;
            text-align: center;
            font-size: 30px;
        }

        .card {
            margin-bottom: 24px;
            padding: 25px;
            border: 1px solid #e5e7eb;
            border-radius: 14px;
            background: #ffffff;

            box-shadow:
                0 6px 20px
                rgba(0, 0, 0, 0.06);
        }

        h2 {
            margin: 0 0 18px;
            color: #111827;
            font-size: 20px;
        }

        .upload-box {
            min-height: 220px;
            padding: 20px;
            display: grid;
            place-items: center;
            overflow: hidden;
            border: 2px dashed #93c5fd;
            border-radius: 12px;
            background: #f8fbff;
            text-align: center;
            cursor: pointer;
            transition: 0.2s;
        }

        .upload-box:hover,
        .upload-box.dragging {
            border-color: #2563eb;
            background: #eff6ff;
        }

        .upload-text strong {
            display: block;
            color: #1f2937;
            font-size: 17px;
        }

        .upload-text span {
            display: block;
            margin-top: 8px;
            color: #6b7280;
            font-size: 14px;
        }

        #uploadPreview {
            max-width: 100%;
            max-height: 200px;
            object-fit: contain;
            border-radius: 8px;
        }

        #predictButton {
            width: 100%;
            margin-top: 15px;
            padding: 14px;
            border: none;
            border-radius: 9px;
            color: white;
            background: #2563eb;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            transition: 0.2s;
        }

        #predictButton:hover:not(:disabled) {
            background: #1d4ed8;
        }

        #predictButton:disabled {
            opacity: 0.45;
            cursor: not-allowed;
        }

        #resultSection {
            display: none;
        }

        #resultSection.show {
            display: block;
        }

        .result-content {
            display: grid;
            grid-template-columns: 280px 1fr;
            align-items: center;
            gap: 30px;
        }

        #resultImage {
            width: 100%;
            height: 240px;
            padding: 8px;
            object-fit: contain;
            border: 1px solid #e5e7eb;
            border-radius: 10px;
            background: #f9fafb;
        }

        .result-label {
            margin: 0 0 8px;
            color: #6b7280;
            font-size: 14px;
        }

        #predictedLetter {
            margin: 0;
            color: #2563eb;
            font-size: 36px;
            font-weight: bold;
        }

        #confidence {
            display: inline-block;
            margin: 18px 0 0;
            padding: 10px 15px;
            border-radius: 8px;
            color: #047857;
            background: #d1fae5;
            font-size: 16px;
            font-weight: bold;
        }

        #confidence.rejected {
            color: #b45309;
            background: #fef3c7;
        }

        #predictedLetter.rejected {
            color: #b45309;
            font-size: 27px;
        }

        #errorMessage {
            min-height: 20px;
            margin: 13px 0 0;
            color: #dc2626;
            text-align: center;
            font-size: 14px;
            font-weight: bold;
        }

        @media (max-width: 650px) {
            body {
                padding: 25px 14px;
            }

            h1 {
                font-size: 25px;
            }

            .card {
                padding: 18px;
            }

            .upload-box {
                min-height: 200px;
            }

            .result-content {
                grid-template-columns: 1fr;
                gap: 22px;
                text-align: center;
            }

            #resultImage {
                height: 210px;
            }
        }
    </style>
</head>

<body>
    <main class="container">

        <h1>Sylheti Nagri Recognizer</h1>

        <section class="card">
            <h2>Upload Image</h2>

            <label
                class="upload-box"
                id="uploadBox"
            >
                <input
                    type="file"
                    id="imageInput"
                    accept="image/png,image/jpeg"
                    hidden
                >

                <div
                    id="uploadText"
                    class="upload-text"
                >
                    <strong>
                        Click or drop an image here
                    </strong>

                    <span>
                        PNG or JPG · Maximum 10 MB
                    </span>
                </div>

                <img
                    id="uploadPreview"
                    alt="Uploaded image"
                    hidden
                >
            </label>

            <button
                id="predictButton"
                disabled
            >
                Predict
            </button>

            <p id="errorMessage"></p>
        </section>

        <section
            class="card"
            id="resultSection"
        >
            <h2>Prediction Result</h2>

            <div class="result-content">
                <img
                    id="resultImage"
                    alt="Original uploaded image"
                >

                <div>
                    <p class="result-label">
                        Recognition status
                    </p>

                    <p id="predictedLetter"></p>

                    <p id="confidence"></p>
                </div>
            </div>
        </section>

    </main>

    <script>
        const imageInput =
            document.getElementById("imageInput");

        const uploadBox =
            document.getElementById("uploadBox");

        const uploadText =
            document.getElementById("uploadText");

        const uploadPreview =
            document.getElementById("uploadPreview");

        const predictButton =
            document.getElementById("predictButton");

        const resultSection =
            document.getElementById("resultSection");

        const resultImage =
            document.getElementById("resultImage");

        const predictedLetter =
            document.getElementById("predictedLetter");

        const confidence =
            document.getElementById("confidence");

        const errorMessage =
            document.getElementById("errorMessage");

        let selectedFile = null;
        let originalImageData = "";


        function selectImage(file) {
            if (
                !file ||
                !file.type.startsWith("image/")
            ) {
                errorMessage.textContent =
                    "Please select a valid image.";

                return;
            }

            if (
                file.size >
                10 * 1024 * 1024
            ) {
                errorMessage.textContent =
                    "Image size cannot exceed 10 MB.";

                return;
            }

            selectedFile = file;

            const reader = new FileReader();

            reader.onload = function (event) {
                originalImageData =
                    event.target.result;

                uploadPreview.src =
                    originalImageData;

                uploadPreview.hidden = false;
                uploadText.hidden = true;
                predictButton.disabled = false;
            };

            reader.readAsDataURL(file);

            errorMessage.textContent = "";
        }


        function resetUploadField() {
            selectedFile = null;
            imageInput.value = "";

            uploadPreview.removeAttribute(
                "src"
            );

            uploadPreview.hidden = true;
            uploadText.hidden = false;
            predictButton.disabled = true;
        }


        imageInput.addEventListener(
            "change",
            function () {
                selectImage(
                    imageInput.files[0]
                );
            }
        );


        ["dragenter", "dragover"].forEach(
            function (eventName) {
                uploadBox.addEventListener(
                    eventName,
                    function (event) {
                        event.preventDefault();

                        uploadBox.classList.add(
                            "dragging"
                        );
                    }
                );
            }
        );


        ["dragleave", "drop"].forEach(
            function (eventName) {
                uploadBox.addEventListener(
                    eventName,
                    function (event) {
                        event.preventDefault();

                        uploadBox.classList.remove(
                            "dragging"
                        );
                    }
                );
            }
        );


        uploadBox.addEventListener(
            "drop",
            function (event) {
                selectImage(
                    event.dataTransfer.files[0]
                );
            }
        );


        predictButton.addEventListener(
            "click",
            async function () {
                if (!selectedFile) {
                    return;
                }

                const formData =
                    new FormData();

                formData.append(
                    "image",
                    selectedFile
                );

                const originalImage =
                    originalImageData;

                predictButton.disabled = true;
                predictButton.textContent =
                    "Processing...";

                errorMessage.textContent = "";

                try {
                    const response = await fetch(
                        "/predict",
                        {
                            method: "POST",
                            body: formData
                        }
                    );

                    const data =
                        await response.json();

                    if (!response.ok) {
                        throw new Error(
                            data.error ||
                            "Prediction failed."
                        );
                    }

                    resultImage.src =
                        originalImage;

                    predictedLetter.classList.remove(
                        "rejected"
                    );

                    confidence.classList.remove(
                        "rejected"
                    );

                    if (data.recognized) {
                        predictedLetter.textContent =
                            `${data.class_number} - ${data.letter}`;

                        confidence.textContent =
                            `Confidence: ${data.confidence.toFixed(2)}%`;

                    } else {
                        predictedLetter.textContent =
                            "Unable to recognize";

                        confidence.textContent =
                            "Please upload a clearer image";

                        predictedLetter.classList.add(
                            "rejected"
                        );

                        confidence.classList.add(
                            "rejected"
                        );
                    }

                    resultSection.classList.add(
                        "show"
                    );

                    resetUploadField();

                    resultSection.scrollIntoView({
                        behavior: "smooth",
                        block: "start"
                    });

                } catch (error) {
                    errorMessage.textContent =
                        error.message;

                    predictButton.disabled =
                        false;

                } finally {
                    predictButton.textContent =
                        "Predict";
                }
            }
        );
    </script>
</body>
</html>
"""


def get_model():
    global model

    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model file not found: {MODEL_PATH}"
            )

        model = load_model(
            MODEL_PATH,
            compile=False
        )

    return model


def preprocess_image(file_bytes, loaded_model):
    image_data = np.frombuffer(
        file_bytes,
        dtype=np.uint8
    )

    image = cv2.imdecode(
        image_data,
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise ValueError(
            "Could not read the uploaded image."
        )

    input_shape = loaded_model.input_shape

    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    image_height = input_shape[1]
    image_width = input_shape[2]
    channels = input_shape[3]

    background = cv2.GaussianBlur(
        image,
        (0, 0),
        sigmaX=15,
        sigmaY=15
    )

    cleaned_image = cv2.divide(
        image,
        background,
        scale=255
    )

    if channels == 1:
        cleaned_image = cv2.cvtColor(
            cleaned_image,
            cv2.COLOR_BGR2GRAY
        )

    elif channels == 3:
        cleaned_image = cv2.cvtColor(
            cleaned_image,
            cv2.COLOR_BGR2RGB
        )

    else:
        raise ValueError(
            f"Unsupported model input shape: "
            f"{input_shape}"
        )

    resized_image = cv2.resize(
        cleaned_image,
        (image_width, image_height),
        interpolation=cv2.INTER_AREA
    )

    image_input = (
        resized_image.astype("float32") /
        255.0
    )

    if channels == 1:
        image_input = np.expand_dims(
            image_input,
            axis=-1
        )

    image_input = np.expand_dims(
        image_input,
        axis=0
    )

    return image_input


@app.route("/")
def home():
    return render_template_string(HTML)


@app.route(
    "/predict",
    methods=["POST"]
)
def predict():
    if "image" not in request.files:
        return jsonify({
            "error": "Please upload an image."
        }), 400

    try:
        uploaded_image = request.files["image"]
        loaded_model = get_model()

        image_input = preprocess_image(
            uploaded_image.read(),
            loaded_model
        )

        prediction = loaded_model.predict(
            image_input,
            verbose=0
        )[0]

        if len(prediction) != len(data_map):
            raise ValueError(
                f"Model output has "
                f"{len(prediction)} classes, "
                f"but data_map has "
                f"{len(data_map)}."
            )

        predicted_index = int(
            np.argmax(prediction)
        )

        predicted_class = predicted_index + 1

        confidence_score = float(
            prediction[predicted_index]
        ) * 100

        if confidence_score < MIN_CONFIDENCE:
            return jsonify({
                "recognized": False
            })

        return jsonify({
            "recognized": True,
            "class_number": predicted_class,
            "letter": data_map[predicted_class],
            "confidence": confidence_score
        })

    except Exception as error:
        return jsonify({
            "error": str(error)
        }), 400


@app.errorhandler(413)
def file_too_large(_error):
    return jsonify({
        "error": (
            "Image is too large. "
            "Maximum size is 10 MB."
        )
    }), 413


if __name__ == "__main__":
    app.run(debug=True)