import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="🩺 Retinal Disease Classification",
    page_icon="👁️",
    layout="wide"
)

# -----------------------------
# Custom CSS with animated gradient and hover effects
# -----------------------------
st.markdown("""
<style>
/* Animated gradient background */
@keyframes gradientBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

.stApp {
    background: linear-gradient(-45deg, #e0c3fc, #8ec5fc, #fcd3fc, #d8b4fe);
    background-size: 400% 400%;
    animation: gradientBG 15s ease infinite;
    color: #1e293b;
    max-width: 900px;
    margin: auto;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    padding: 20px 40px;
    transition: background 0.5s ease;
}

/* Header */
.custom-header {
    color: #7c3aed; /* purple */
    text-align: center;
    font-size: 50px;
    font-weight: 700;
    margin-bottom: 5px;
}

/* Subtitle */
.custom-subtitle {
    color: #d8b4fe; /* light purple / magenta */
    text-align: center;
    font-size: 22px;
    font-weight: 500;
    margin-top: 0;
    margin-bottom: 30px;
}

/* File uploader with gradient and shadow */
.css-1v0mbdj.edgvbvh3 {
    border: none;
    border-radius: 15px;
    padding: 25px;
    background: linear-gradient(to right, #fcd3fc, #d8b4fe);
    color: #1e293b;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.css-1v0mbdj.edgvbvh3:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 25px rgba(0,0,0,0.2);
}

/* Image display */
.stImage {
    border-radius: 15px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.08);
}

/* Prediction card with hover effect */
.prediction-card {
    background: linear-gradient(to right, #e0c3fc, #8ec5fc);
    padding: 20px;
    border-radius: 20px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    text-align: center;
    font-size: 24px;
    font-weight: 600;
    color: #1e3a8a; /* dark text */
    margin-top: 20px;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.prediction-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 25px rgba(0,0,0,0.2);
}

/* Progress bar */
.stProgress > div > div > div {
    background-color: #7c3aed !important; /* purple */
}

/* Info box */
.stInfo {
    font-size: 18px;
    background-color: #ede9fe !important; /* pastel light purple */
    color: #7c3aed !important;
    border-radius: 12px;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header & Subtitle in HTML
# -----------------------------
st.markdown('<div class="custom-header">🩺 Retinal Disease Classification System</div>', unsafe_allow_html=True)
st.markdown('<div class="custom-subtitle">Upload a retinal image to detect possible eye diseases using CNN (MobileNetV2)</div>', unsafe_allow_html=True)

# -----------------------------
# File uploader
# -----------------------------
uploaded_file = st.file_uploader(
    "📤 Drag & drop a retinal image here or click to browse (JPG, PNG, max 200MB)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Retinal Image", use_column_width=True)

    st.info("👁️ Running model inference...")

    # -----------------------------
    # Load tflite model
    # -----------------------------
    interpreter = tf.lite.Interpreter(model_path="mobilenetv2_eye_disease.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Preprocess image
    img = image.resize((224, 224))
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]

    # Predicted class and confidence
    classes = ["Normal", "Diabetic Retinopathy", "Glaucoma", "AMD"]
    pred_class = np.argmax(output_data)
    confidence = output_data[pred_class]

    # Display results in a card
    st.markdown(f"""
        <div class="prediction-card">
            ✅ Predicted Disease: <b>{classes[pred_class]}</b><br>
            Confidence: {confidence*100:.2f}%
        </div>
    """, unsafe_allow_html=True)

else:
    st.info("👁️ Please upload a retinal image to begin diagnosis.")

