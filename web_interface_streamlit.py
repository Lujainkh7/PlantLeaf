import base64
import numpy as np
import streamlit as st
import tensorflow as tf
from streamlit_option_menu import option_menu
from PIL import Image, ImageFilter
from pathlib import Path

# ✅ إعداد الصفحة
st.set_page_config(page_title="PlantLeaf", layout="centered")

# ✅ اختيار اللغة
language = st.selectbox("\U0001F310 Select Language | اختر اللغة", ["English", "العربية"])

# ✅ إعداد الخلفية

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

# ✅ تحميل النموذج وأسماء الأصناف
img_height = 224
img_width = 224
model = tf.keras.models.load_model("mobilenetv2_transfer_model.h5")
class_names = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry_(including_sour)___healthy", "Cherry_(including_sour)___Powdery_mildew",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_", "Corn_(maize)___healthy",
    "Corn_(maize)___Northern_Leaf_Blight", "Grape___Black_rot", "Grape___Esca_(Black_Measles)", "Grape___healthy",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot",
    "Peach___healthy", "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight",
    "Potato___healthy", "Potato___Late_blight", "Raspberry___healthy", "Soybean___healthy",
    "Squash___Powdery_mildew", "Strawberry___healthy", "Strawberry___Leaf_scorch", "Tomato___Bacterial_spot",
    "Tomato___Early_blight", "Tomato___healthy", "Tomato___Late_blight", "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
    "Tomato___Tomato_mosaic_virus", "Tomato___Tomato_Yellow_Leaf_Curl_Virus"
]

# ✅ القوائم حسب اللغة
if language == "English":
    menu_options = ["plantLeaf", "About", "Team", "Contact", "My Profile"]

    menu_labels = {
        "upload": "Choose a leaf image...",
        "title": "🌿Plant and Leaf Disease Classification",
        "desc": "Upload a leaf image and we'll identify the disease with high accuracy.",
        "diagnosis": "Diagnosis:",
        "about": "PlantLeaf is a simple web app that helps you find out if your plant is sick.Just upload a picture of a leaf, and the app will tell you the disease name in seconds.It’s fast, easy to use, and supports many types of plants.",
        "team": " This project was developed by **Lujain Kharboutli**, a student of Information Technology Engineering - Intelligent Systems,at the Syrian Virtual University.",
        "contact": ["Email", "Mobile", "Location"],
          }
else:
    menu_options = ["الرئيسية", "حول", "الفريق", "اتصل بنا", "الملف الشخصي"]
    menu_labels = {
        "upload": "اختر صورة ورقة...",
        "title": "🌿 تصنيف أمراض النبات",
        "desc": "قم برفع صورة لورقة النبات وسنقوم بتحديد المرض بدقة عالية.",
        "diagnosis": "نتيجة التشخيص:",
        "about": "PlantLeaf هو تطبيق بسيط يساعدك على معرفة ما إذا كانت نبتتك مريضة.كل ما عليك هو رفع صورة للورقة، وسيعرض لك اسم المرض خلال ثوانٍ.التطبيق سريع وسهل الاستخدام، ويدعم أنواعًا كثيرة من النباتات.",
        "team": "تم تطوير هذا المشروع من قبل **لجين حربوطلي** طالبة في الجامعة الافتراضية السورية في هندسة تكنولوجيا المعلومات قسم الذكاء الصنعي ",
        "contact": ["البريد الإلكتروني", "اتصل", "الموقع"],
    }

selected = option_menu(
    menu_title=None,
    options=menu_options,
    icons=["herbs", "info-circle", "people", "envelope"],
    default_index=0,
    orientation="horizontal"
)

# ✅ الصفحة الرئيسية
if selected in ["plantLeaf", "الرئيسية"]:
    st.markdown(f"<h2 style='color:#2e7d32;'>{menu_labels['title']}</h2>", unsafe_allow_html=True)
    st.write(menu_labels["desc"])

    uploaded_file = st.file_uploader(menu_labels["upload"], type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        sharpened_image = image.filter(ImageFilter.SHARPEN)
        st.image(sharpened_image, caption="Uploaded Image", width=500)

        img = image.resize((224, 224))
        img_array = tf.keras.utils.img_to_array(img) / 255.0
        img_array = tf.expand_dims(img_array, axis=0)


        predictions = model.predict(img_array)


        score = tf.nn.softmax(predictions[0])
        predicted_class = class_names[np.argmax(score)]
        confidence = 100 * np.max(score)

        st.success(f"{menu_labels['diagnosis']} {predicted_class}")

elif selected in ["About", "حول"]:
    st.subheader(menu_options[1])
    st.write(menu_labels["about"])

elif selected in ["Team", "الفريق"]:
    st.subheader(menu_options[2])
    st.write(menu_labels["team"])

elif selected in ["Contact", "اتصل بنا"]:
    st.subheader(menu_options[3])
    if language == "English":
        st.write(" Email: lougainkh@gmail.com")
        st.write(" Mobile: +963 958 *** ***")
        st.write(" Location: Syria")
    else:
        st.write(" البريد الإلكتروني: lougainkh@gmail.com")
        st.write(" الهاتف: +963 958 *** ***")
        st.write(" الموقع: سوريا")

elif selected in ["My Profile", "الملف الشخصي"]:
    st.title(menu_options[4])
    tab = st.radio("اختر إجراء" if language == "العربية" else "Choose an action:", ["Sign In", "Register"], horizontal=True)

    if tab == "Sign In":
        username = st.text_input("Username" if language == "English" else "اسم المستخدم")
        password = st.text_input("Password" if language == "English" else "كلمة المرور", type="password")
        if st.button("Login" if language == "English" else "تسجيل الدخول"):
            if username == "admin" and password == "1234":
                st.success("✅ Logged in successfully!" if language == "English" else "✅ تم تسجيل الدخول بنجاح!")
            else:
                st.error("❌ Invalid credentials" if language == "English" else "❌ بيانات الدخول غير صحيحة")

    elif tab == "Register":
        new_username = st.text_input("New Username" if language == "English" else "اسم مستخدم جديد")
        new_password = st.text_input("New Password" if language == "English" else "كلمة مرور جديدة", type="password")
        confirm_password = st.text_input("Confirm Password" if language == "English" else "تأكيد كلمة المرور", type="password")
        if st.button("Create Account" if language == "English" else "إنشاء الحساب"):
            if new_password == confirm_password:
                st.success(f"Account created for {new_username}" if language == "English" else f"تم إنشاء الحساب لـ {new_username}")
            else:
                st.error("❌ Passwords do not match" if language == "English" else "❌ كلمات المرور غير متطابقة")

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
    .footer a {
        color: #2e7d32;
        text-decoration: none;
        margin: 0 5px;
    }
    .footer a:hover {
        text-decoration: underline;
        color: #388e3c;
    }
    </style>

    <div class="footer">
        🌿 <a href="#Home">Home</a> | <a href="#About">About</a> | <a href="#Team">Team</a> | <a href="#Contact">Contact</a><br>
        © 2025 PlantLeaf | Developed by Lujain Kharboutli 
    </div>
""", unsafe_allow_html=True)

# .\env_plantleaf\Scripts\activate
# streamlit run web_interface_streamlit.py
