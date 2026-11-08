from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps, UnidentifiedImageError


IMG_SIZE = 224
MODEL_PATH = Path(__file__).resolve().with_name("model_lalua.h5")
CLASS_NAMES = [
    "Bacterial Leaf Blight",
    "Brown Spot",
    "Healthy Rice Leaf",
    "Leaf Blast",
    "Leaf scald",
    "Sheath Blight",
]
DISPLAY_NAMES = {
    "Bacterial Leaf Blight": "Bệnh bạc lá",
    "Brown Spot": "Bệnh đốm nâu",
    "Healthy Rice Leaf": "Lá lúa khỏe mạnh",
    "Leaf Blast": "Bệnh đạo ôn",
    "Leaf scald": "Bệnh cháy lá",
    "Sheath Blight": "Bệnh khô vằn",
}

st.set_page_config(page_title="Nhận diện bệnh lá lúa bằng AI", layout="centered")
st.title("Hệ thống nhận diện bệnh lá lúa")
st.write("Tải ảnh lá lúa lên để nhận dự đoán tham khảo từ mô hình AI.")


@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    if model.input_shape != (None, IMG_SIZE, IMG_SIZE, 3):
        raise ValueError(f"Kích thước đầu vào không đúng: {model.input_shape}")
    if model.output_shape != (None, len(CLASS_NAMES)):
        raise ValueError(f"Số lớp đầu ra không đúng: {model.output_shape}")
    return model


if not MODEL_PATH.is_file():
    st.error(f"Không tìm thấy model: {MODEL_PATH.name}. Hãy đặt model cạnh app.py.")
    st.stop()

try:
    model = load_model()
except (OSError, ValueError) as exc:
    st.error(f"Không thể nạp model {MODEL_PATH.name}: {exc}")
    st.stop()


def make_gradcam_heatmap(img_array, model, class_index, last_conv_layer_name="Conv_1"):
    grad_model = tf.keras.models.Model(
        model.input,
        [model.get_layer(last_conv_layer_name).output, model.output],
    )
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_array)
        class_score = predictions[:, class_index]

    gradients = tape.gradient(class_score, conv_output)
    pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_gradients[..., tf.newaxis]
    heatmap = tf.maximum(tf.squeeze(heatmap), 0)
    heatmap = tf.math.divide_no_nan(heatmap, tf.reduce_max(heatmap))
    return heatmap.numpy()


uploaded_file = st.file_uploader(
    "Chọn ảnh lá lúa (JPG, JPEG hoặc PNG)",
    type=["jpg", "jpeg", "png"],
    max_upload_size=10,
)

if uploaded_file is not None:
    try:
        with Image.open(uploaded_file) as uploaded_image:
            image = ImageOps.exif_transpose(uploaded_image).convert("RGB")
    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombError,
    ) as exc:
        st.error(f"Không thể đọc ảnh đã tải lên: {exc}")
        st.stop()

    st.image(
        image,
        caption="Ảnh đã tải lên",
        width="stretch",
        alt="Ảnh lá lúa do người dùng tải lên",
    )

    resized_image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.expand_dims(
        np.asarray(resized_image, dtype=np.float32), axis=0
    ) / 255.0
    predictions = model.predict(img_array, verbose=0)[0]

    if predictions.shape != (len(CLASS_NAMES),) or not np.isfinite(predictions).all():
        st.error("Model trả về kết quả không hợp lệ. Hãy kiểm tra lại file model.")
        st.stop()

    predicted_index = int(np.argmax(predictions))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(predictions[predicted_index])

    st.info(
        f"Dự đoán tham khảo: **{DISPLAY_NAMES[predicted_class]}**",
        title="Kết quả AI",
    )
    st.metric("Điểm dự đoán (chưa hiệu chuẩn)", f"{confidence:.1%}")
    st.warning(
        "Mô hình luôn chọn một trong 6 nhãn đã học, kể cả khi ảnh không phải lá lúa. "
        "Kết quả chỉ mang tính tham khảo; hãy kiểm tra thực tế trước khi xử lý cây trồng."
    )

    st.subheader("Vùng ảnh mô hình tập trung (Grad-CAM)")
    try:
        heatmap = make_gradcam_heatmap(img_array, model, predicted_index)
    except (KeyError, ValueError, TypeError, tf.errors.InvalidArgumentError) as exc:
        st.warning(f"Không thể tạo Grad-CAM: {type(exc).__name__}: {exc}")
    else:
        if not np.isfinite(heatmap).all() or not np.any(heatmap):
            st.warning("Mô hình không tạo được bản đồ nhiệt có ý nghĩa cho ảnh này.")
        else:
            image_array = np.asarray(resized_image)
            image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
            heatmap_colored = cv2.applyColorMap(
                np.uint8(255 * heatmap), cv2.COLORMAP_JET
            )
            overlay = cv2.addWeighted(image_bgr, 0.6, heatmap_colored, 0.4, 0)
            overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            st.image(
                overlay,
                caption="Vùng mô hình tập trung khi đưa ra dự đoán",
                width="stretch",
                alt="Bản đồ nhiệt Grad-CAM phủ trên ảnh lá lúa",
            )
