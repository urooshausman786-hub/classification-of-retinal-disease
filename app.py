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
# Custom CSS for modern clean UI
# -----------------------------
st.markdown("""
<style>
/* Light gradient background */
.stApp {
    background: linear-gradient(to bottom, #f0f4f8, #d9e2ec);
    color: #1f2937; /* Dark text for readability */
    max-width: 900px;
    margin: auto;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    padding: 20px 40px;
}

/* Header */
h1 {
    color: #0f172a;
    text-align: center;
    font-size: 48px;
    font-weight: 700;
    margin-bottom: 5px;
}

/* Subtitle */
h3 {
    color: #334155;
    text-align: center;
    font-size: 22px;
    font-weight: 500;
    margin-top: 0;
    margin-bottom: 30px;
}

/* File uploader */
.css-1v0mbdj.edgvbvh3 {
    border: 2px dashed #94a3b8;
    border-radius: 12px;
    padding: 25px;
    background-color: rgba(255, 255, 255, 0.8);
    color: #1f2937;
}

/* Image display */
.stImage {
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}

/* Prediction result */
.prediction {
    font-size: 26px;
    font-weight: 600;
    color: #0f172a;
}

/* Confidence bar */
.stProgress > div > div > div {
    background-color: #2563eb !important; /* Blue progress bar */
}

/* Success box */
.stSuccess {
    font-size: 24px;
    font-weight: 600;
    background-color: #d1fae5 !important; /* Light green background */
    color: #065f46 !important; /* Dark green text */
    border-radius: 8px;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.title("🩺 Retinal Disease Classification System")
st.markdown("<h3>Upload a retinal image to detect possible eye diseases using CNN (MobileNetV2)</h3>", unsafe_allow_html=True)

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

    # Display results in columns
    col1, col2 = st.columns([2,1])
    with col1:
        st.success(f"✅ Predicted Disease: **{classes[pred_class]}**")
    with col2:
        st.progress(int(confidence*100))
        st.write(f"Confidence: {confidence*100:.2f}%")

else:
    st.info("👁️ Please upload a retinal image to begin diagnosis.")
