import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# 1. Cấu hình giao diện web
st.set_page_config(page_title="AI Chẩn Đoán Bệnh Lá Lúa", layout="centered")
st.title("🌾 Hệ Thống AI Chẩn Đoán Bệnh Lá Lúa")
st.write("Vui lòng tải ảnh lá lúa lên để hệ thống phân tích.")

# 2. Nạp "bộ não" AI đã huấn luyện
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('model_lalua.h5')

model = load_model()

# Khai báo đúng 6 loại bệnh theo thư mục của bạn
CLASS_NAMES = ['Bacterial Leaf Blight', 'Brown Spot', 'Healthy Rice Leaf', 'Leaf Blast', 'Leaf scald', 'Sheath Blight']

# 3. Hàm tạo bản đồ nhiệt Grad-CAM (Giải thích AI)
def make_gradcam_heatmap(img_array, model, last_conv_layer_name="Conv_1"):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# 4. Chức năng tải ảnh lên và phân tích
uploaded_file = st.file_uploader("Chọn một bức ảnh (JPG, PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Mở ảnh và ép về chuẩn màu RGB (Tránh lỗi 4 kênh màu)
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Ảnh lá lúa tải lên', use_container_width=True)
    st.write("🔄 Hệ thống đang phân tích...")

    # Tiền xử lý ảnh
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Chuẩn hóa pixel

    # Dự đoán
    predictions = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0]) * 100

    # In kết quả chẩn đoán
    st.success(f"**Kết quả chẩn đoán:** Bệnh {predicted_class}")
    st.info(f"**Độ tin cậy của AI:** {confidence:.2f}%")

    # Hiển thị Grad-CAM
    st.markdown("### 🔍 Giải thích từ AI (Grad-CAM)")
    try:
        heatmap = make_gradcam_heatmap(img_array, model)
        
        # Chồng bản đồ nhiệt lên ảnh gốc
        img_cv = cv2.cvtColor(np.array(image.resize((224, 224))), cv2.COLOR_RGB2BGR)
        heatmap_resized = cv2.resize(heatmap, (img_cv.shape[1], img_cv.shape[0]))
        heatmap_colored = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(img_cv, 0.6, heatmap_colored, 0.4, 0)
        superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
        
        st.image(superimposed_img, caption="Vùng AI tập trung phân tích để đưa ra kết luận", use_container_width=True)
    except Exception as e:
        st.warning("Mô hình hiện tại không hỗ trợ trích xuất lớp tích chập cuối cùng để vẽ Grad-CAM.")