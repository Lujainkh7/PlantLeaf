import base64
import numpy as np
import streamlit as st
import tensorflow as tf
from streamlit_option_menu import option_menu
from PIL import Image, ImageFilter
from pathlib import Path


def set_background(image_path):
    img_path = Path(image_path)
    img_bytes = img_path.read_bytes()
    encoded = base64.b64encode(img_bytes).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
        )

set_background("PlantLeafbackground.png")

# إعدادات النموذج
img_height = 224
img_width = 224
model = tf.keras.models.load_model("mobilenetv2_transfer_model.h5")
class_names = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___healthy",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___healthy",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___healthy",
    "Potato___Late_blight",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___healthy",
    "Strawberry___Leaf_scorch",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___healthy",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus"
]


st.set_page_config(page_title="PlantLeaf", layout="centered")

# 🌿 شريط التنقل
selected = option_menu(
    menu_title=None,
    options=["plantLeaf", "About", "Team", "Contact", "My Profile"],
    icons=["herbs", "info-circle", "people", "envelope"],
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#e8f5e9"},
        "icon": {"color": "#2e7d32", "font-size": "18px"},
        "nav-link": {"font-size": "16px", "text-align": "center"},
        "nav-link-selected": {"background-color": "#66bb6a", "color": "white"},
    }
)

# 🌿 الصفحة الرئيسية
if selected == "plantLeaf":
    st.markdown("<h2 style='color:#2e7d32;'> Intelligent Plant Disease Classifier</h2>", unsafe_allow_html=True)
    st.write(" Upload a leaf image and we'll identify the disease with high accuracy.")

    uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")  # ✅ تأكد من 3 قنوات
        sharpened_image = image.filter(ImageFilter.SHARPEN)
        st.image(sharpened_image, caption=" Uploaded Image", width=500)

        img = image.resize((224, 224))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = img_array / 255.0     # ⚠️ Normalize manually مثل ما كان بالتدريب
        img_array = tf.expand_dims(img_array, axis=0)

       # predictions = model.predict(img_array)
        #score = tf.nn.softmax(predictions[0])
        #predicted_class = class_names[np.argmax(score)]
        #confidence = 100 * np.max(score)
        predictions = model.predict(img_array)
        print("Raw predictions:", predictions)

        score = tf.nn.softmax(predictions[0])
        print("Softmax:", score.numpy())

        predicted_index = np.argmax(score)
        predicted_class = class_names[predicted_index]
        confidence = 100 * np.max(score)

        st.success(f"Diagnosis:  {predicted_class}")
        st.info(f"Confidence: {confidence:.2f}%")

       # st.success(f"Diagnosis:  {predicted_class}")
        # st.info(f"Confidence: {confidence:.2f}%")

elif selected == "About":
    st.subheader("About PlantLeaf")
    st.write("""
    PlantLeaf is an advanced AI-powered system designed to detect and classify plant diseases from leaf images.
    It uses deep learning (CNN models) trained on the PlantVillage dataset to provide fast and accurate diagnosis.
    The project aims to support farmers and researchers in early disease detection and reducing crop losses.
    """)

elif selected == "Team":
    st.subheader("")
    st.write("""
    This project was developed by **Lujain Kharboutli**, a student of Information Technology Engineering - Intelligent Systems,
    at the Syrian Virtual University.
    """)

elif selected == "Contact":
    st.subheader("Contact Us")
    st.write("📧 Email: lougainkh@gmail.com")
    st.write("📱 Mobile: +963 958 *** ***")
    st.write("📍 Location: Syria")

elif selected == "My Profile":
    st.title(" My Profile")
    tab = st.radio("Choose an action:", ["Sign In", "Register"], horizontal=True)

    if tab == "Sign In":
        st.subheader(" Sign In")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if username == "admin" and password == "1234":
                st.success("✅ Logged in successfully!")
            else:
                st.error("❌ Invalid credentials")

    elif tab == "Register":
        st.subheader(" Register")
        new_username = st.text_input("New Username")
        new_password = st.text_input("New Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        if st.button("Create Account"):
            if new_password == confirm_password:
                st.success(f" Account created for {new_username}")
            else:
                st.error("❌ Passwords do not match")

# ----- 🔻 شريط سفلي -----
st.markdown("""
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #e0f2f1;
        text-align: center;
        padding: 10px;
        color: #2e7d32;
        font-size: 13px;
        box-shadow: 0 -1px 3px rgba(0,0,0,0.1);
    }
    </style>

    <div class="footer">
        🌿 <a href="#Home">Home</a> | <a href="#About">About</a> | <a href="#Team">Team</a> | <a href="#Contact">Contact</a><br>
        © 2025 PlantLeaf | Developed by Lujain Kharboutli 
    </div>
""", unsafe_allow_html=True)

# .\env_plantleaf\Scripts\activate
# streamlit run web_interface_streamlit.py
