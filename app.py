import os
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from google import genai

# Thử nạp các module có sẵn trong dự án của bạn (nếu có hàm dùng chung)
try:
    import preprocess
except ImportError:
    preprocess = None

try:
    import weather_service
except ImportError:
    weather_service = None

# 1. Cấu hình trang
st.set_page_config(
    page_title="Trợ lý AI Chuẩn đoán Bệnh Lá Lúa",
    page_icon="🌾",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_lalua.h5")

# Danh sách nhãn bệnh phổ biến trên lá lúa (bạn chỉnh lại thứ tự đúng với lúc train model_lalua.h5 nhé)
CLASS_NAMES = [
    "Bệnh đạo ôn (Rice Blast)",
    "Bệnh bạc lá / cháy bìa lá (Bacterial Leaf Blight)",
    "Bệnh đốm nâu (Brown Spot)",
    "Bệnh khô vằn (Sheath Blight)",
    "Lá lúa khỏe mạnh (Healthy)"
]

# 2. Hàm tải mô hình Deep Learning
@st.cache_resource
def load_rice_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)

# 3. Hàm tiền xử lý ảnh đầu vào khớp với kích thước của model
def prepare_image(image: Image.Image, target_size=(224, 224)):
    img = image.convert("RGB").resize(target_size)
    img_array = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

# 4. Hàm gọi Gemini AI tư vấn phác đồ điều trị
def get_ai_treatment_advice(disease_name: str, confidence: float, weather_info: str, api_key: str):
    if not api_key:
        return "⚠️ Vui lòng nhập **Gemini API Key** ở thanh bên trái để nhận tư vấn phác đồ điều trị từ trợ lý AI."
    try:
        client = genai.Client(api_key=api_key)
        prompt = (
            f"Bạn là chuyên gia nông nghiệp về cây lúa tại Việt Nam.\n"
            f"- Kết quả nhận diện hình ảnh lá lúa: **{disease_name}** (độ tin cậy {confidence:.1f}%).\n"
            f"- Thông tin thời tiết khu vực: {weather_info if weather_info else 'Chưa cung cấp'}.\n\n"
            "Hãy tư vấn ngắn gọn, rõ ràng cho bà con nông dân gồm:\n"
            "1. Nhận diện triệu chứng đặc trưng của bệnh này để đối chiếu.\n"
            "2. Nguyên nhân phát sinh và mức độ ảnh hưởng của điều kiện thời tiết hiện tại.\n"
            "3. Biện pháp canh tác & phòng trừ (ưu tiên an toàn sinh học, kèm hoạt chất thuốc BVTV khuyến cáo nếu bệnh nặng)."
        )
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return response.text
    except Exception as e:
        return f"Lỗi khi gọi trợ lý AI Gemini: {e}"

# ================= GIAO DIỆN CHÍNH =================
st.title("🌾 Trợ lý AI Chuẩn đoán Bệnh qua Hình ảnh Lá Lúa")
st.caption("Kết hợp mô hình Thị giác máy tính (CNN) và Trợ lý AI tư vấn canh tác thông minh")

# Sidebar cấu hình
with st.sidebar:
    st.header("⚙️ Cấu hình hệ thống")
    api_key = st.text_input(
        "Gemini API Key",
        value=os.getenv("GEMINI_API_KEY", ""),
        type="password",
        help="Nhập API key từ Google AI Studio để kích hoạt tư vấn điều trị."
    )
    city = st.text_input("Khu vực canh tác (Tỉnh/Thành phố)", value="Cần Thơ")
    st.divider()
    st.info("📌 **Hướng dẫn:** Tải lên ảnh chụp rõ nét vùng lá lúa có vết bệnh để hệ thống phân tích.")

# Tải model
model = load_rice_model()
if model is None:
    st.error(f"Không tìm thấy file mô hình tại `{MODEL_PATH}`.")
    st.stop()

# Tự động lấy kích thước ảnh đầu vào từ kiến trúc model_lalua.h5
input_shape = model.input_shape
img_height = input_shape[1] if len(input_shape) >= 3 and input_shape[1] else 224
img_width = input_shape[2] if len(input_shape) >= 3 and input_shape[2] else 224

# Chia 2 cột giao diện
col_left, col_right = st.columns([1, 1.3], gap="large")

with col_left:
    st.subheader("1. Tải ảnh lá lúa")
    uploaded_file = st.file_uploader("Chọn ảnh lá lúa (JPG, PNG, JPEG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Ảnh lá lúa đã tải lên", use_container_width=True)

with col_right:
    st.subheader("2. Kết quả chuẩn đoán & Tư vấn")
    if uploaded_file is not None:
        with st.spinner("Đang phân tích hình ảnh lá lúa..."):
            processed_img = prepare_image(image, target_size=(img_width, img_height))
            predictions = model.predict(processed_img)[0]
            pred_idx = int(np.argmax(predictions))
            confidence = float(predictions[pred_idx]) * 100

            # Kiểm tra số lượng lớp của model để tránh lỗi lệch index
            if len(predictions) == len(CLASS_NAMES):
                predicted_label = CLASS_NAMES[pred_idx]
            else:
                predicted_label = f"Nhóm bệnh số {pred_idx}"

        if "khỏe mạnh" in predicted_label.lower():
            st.success(f"**Kết quả:** {predicted_label} (Độ tin cậy: **{confidence:.1f}%**)")
        else:
            st.warning(f"**Phát hiện:** {predicted_label} (Độ tin cậy: **{confidence:.1f}%**)")

        # Lấy thông tin thời tiết nếu weather_service có hàm hỗ trợ
        weather_summary = f"Khu vực {city}"
        if weather_service is not None:
            for fn_name in ("get_weather", "get_current_weather", "fetch_weather"):
                if hasattr(weather_service, fn_name):
                    try:
                        w_data = getattr(weather_service, fn_name)(city)
                        weather_summary = str(w_data)
                    except Exception:
                        pass
                    break

        st.markdown("### 📋 Phác đồ xử lý từ Trợ lý AI")
        with st.spinner("Trợ lý AI đang tổng hợp hướng dẫn điều trị..."):
            advice = get_ai_treatment_advice(predicted_label, confidence, weather_summary, api_key)
            st.markdown(advice)
    else:
        st.info("👈 Hãy tải lên một bức ảnh lá lúa ở cột bên trái để bắt đầu chuẩn đoán.")