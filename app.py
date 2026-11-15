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

DISEASE_INFO = {
    "Bacterial Leaf Blight": {
        "symptoms": "Vết úa vàng hoặc xám trắng chạy dọc mép lá, lan dần vào trong; sáng sớm có thể thấy giọt dịch vi khuẩn màu vàng đục.",
        "cause": "Vi khuẩn Xanthomonas oryzae, lây qua nước, vết thương trên lá, hạt giống và tàn dư rơm rạ.",
        "prevention": [
            "Dùng giống kháng bệnh, hạt giống sạch bệnh.",
            "Không bón thừa đạm, chia nhỏ lượng đạm bón.",
            "Vệ sinh đồng ruộng, xử lý rơm rạ và cỏ dại sau thu hoạch.",
            "Hạn chế để nước tràn từ ruộng bệnh sang ruộng khỏe.",
        ],
        "treatment": "Khi bệnh mới xuất hiện, ngưng bón đạm và giữ ruộng thoáng. Có thể dùng thuốc nhóm gốc đồng hoặc kháng sinh nông nghiệp đã đăng ký, theo hướng dẫn của cán bộ bảo vệ thực vật.",
    },
    "Brown Spot": {
        "symptoms": "Đốm nâu hình bầu dục hoặc tròn, tâm xám nhạt, viền nâu sẫm, rải rác trên lá; nặng thì lá khô, hạt lép.",
        "cause": "Nấm Bipolaris oryzae, thường gặp khi ruộng thiếu dinh dưỡng (nhất là kali), đất nghèo hoặc khô hạn.",
        "prevention": [
            "Bón phân cân đối N-P-K, chú ý bổ sung kali và silic.",
            "Xử lý hạt giống trước khi gieo.",
            "Cải tạo đất, đảm bảo đủ nước, tránh để cây bị stress.",
            "Dọn tàn dư cây bệnh sau vụ.",
        ],
        "treatment": "Bổ sung dinh dưỡng trước tiên. Nếu bệnh nặng, có thể dùng thuốc trừ nấm đã đăng ký cho lúa (ví dụ nhóm triazole, strobilurin), theo hướng dẫn trên nhãn thuốc.",
    },
    "Healthy Rice Leaf": {
        "symptoms": "Lá xanh đều, không có vết bệnh rõ rệt.",
        "cause": "Không phát hiện dấu hiệu bệnh trong ảnh này.",
        "prevention": [
            "Thăm đồng thường xuyên, theo dõi sớm các vết lạ trên lá.",
            "Bón phân cân đối, quản lý nước hợp lý.",
            "Giữ ruộng sạch cỏ và tàn dư.",
        ],
        "treatment": "Không cần xử lý. Nên chụp thêm vài lá ở nhiều vị trí khác nhau để chắc chắn.",
    },
    "Leaf Blast": {
        "symptoms": "Vết hình thoi (mắt én), tâm xám trắng, viền nâu; nhiều vết liên kết làm cháy cả lá.",
        "cause": "Nấm Magnaporthe oryzae, phát triển mạnh khi ẩm độ cao, sương mù, đêm lạnh, bón thừa đạm.",
        "prevention": [
            "Dùng giống kháng đạo ôn.",
            "Không bón đạm muộn hoặc quá nhiều.",
            "Sạ cấy mật độ vừa phải để ruộng thoáng.",
            "Theo dõi ruộng kỹ khi thời tiết âm u, sương mù kéo dài.",
        ],
        "treatment": "Phun sớm khi vết bệnh mới xuất hiện. Có thể dùng thuốc đặc trị đạo ôn đã đăng ký (ví dụ tricyclazole, isoprothiolane), theo đúng hướng dẫn và thời gian cách ly.",
    },
    "Leaf scald": {
        "symptoms": "Vết cháy lan từ chóp hoặc mép lá vào trong, có vằn đồng tâm màu nâu và vàng, rìa vết thường loang lổ.",
        "cause": "Nấm Microdochium oryzae, thuận lợi khi ruộng ẩm, bón nhiều đạm, mật độ dày.",
        "prevention": [
            "Sạ cấy mật độ hợp lý, giữ ruộng thoáng.",
            "Bón đạm vừa phải, tăng kali.",
            "Vệ sinh ruộng, xử lý tàn dư sau thu hoạch.",
        ],
        "treatment": "Giảm đạm, giữ nước hợp lý. Nếu bệnh lan rộng, hãy hỏi cán bộ bảo vệ thực vật địa phương để chọn thuốc trừ nấm phù hợp.",
    },
    "Sheath Blight": {
        "symptoms": "Vết loang vằn như da hổ ở bẹ lá sát mặt nước, màu xám lục đến nâu, lan lên các lá phía trên.",
        "cause": "Nấm Rhizoctonia solani, tồn tại trong đất và tàn dư, phát triển mạnh khi sạ dày, ẩm độ cao, bón nhiều đạm.",
        "prevention": [
            "Sạ cấy thưa vừa phải, không để lúa quá rậm.",
            "Bón đạm cân đối, tránh bón dồn một lần.",
            "Dọn sạch rơm rạ, cỏ bờ và hạch nấm nổi trên mặt nước.",
            "Tháo cạn nước hợp lý giữa vụ để giảm ẩm độ gốc.",
        ],
        "treatment": "Có thể dùng thuốc nhóm validamycin hoặc triazole đã đăng ký, phun tập trung vào phần gốc, bẹ lá, theo hướng dẫn trên nhãn.",
    },
}

SEVERITY_NOTE = "Con số này chỉ là ước lượng dựa trên màu sắc, không thay thế đánh giá ngoài đồng."

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

def estimate_severity(rgb_image):
    """Ước lượng % diện tích lá bị bệnh bằng ngưỡng màu HSV (heuristic)."""
    rgb = np.asarray(rgb_image).copy()
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    bgr = cv2.GaussianBlur(bgr, (5, 5), 0)
    h, s, v = cv2.split(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV))

    green = (h >= 35) & (h <= 90) & (s >= 40) & (v >= 40)
    lesion = ((h < 35) | (h > 165)) & (s >= 40) & (v >= 30)

    kernel = np.ones((3, 3), np.uint8)
    lesion = cv2.morphologyEx(lesion.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)
    green = cv2.morphologyEx(green.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)

    leaf_pixels = int(green.sum() + lesion.sum())
    if leaf_pixels < 0.05 * rgb.shape[0] * rgb.shape[1]:
        return None  # không tìm thấy đủ vùng lá

    percent = 100.0 * int(lesion.sum()) / leaf_pixels
    if percent < 5:
        level = "Rất nhẹ"
    elif percent < 20:
        level = "Nhẹ"
    elif percent < 40:
        level = "Trung bình"
    else:
        level = "Nặng"

    overlay = rgb.copy()
    red = np.array([255, 0, 0], dtype=np.float32)
    overlay[lesion] = (0.5 * overlay[lesion] + 0.5 * red).astype(np.uint8)
    return percent, level, overlay

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

    top3_idx = np.argsort(predictions)[::-1][:3]
    predicted_index = int(top3_idx[0])
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(predictions[predicted_index])

    st.info(
        f"Dự đoán tham khảo: **{DISPLAY_NAMES[predicted_class]}**",
        title="Kết quả AI",
    )
    st.metric("Điểm dự đoán (chưa hiệu chuẩn)", f"{confidence:.1%}")

    st.subheader("Top 3 dự đoán")
    for rank, idx in enumerate(top3_idx, start=1):
        prob = float(predictions[idx])
        st.write(f"{rank}. **{DISPLAY_NAMES[CLASS_NAMES[idx]]}**: {prob:.1%}")
        st.progress(min(max(prob, 0.0), 1.0))

    if confidence < 0.6:
        st.warning(
            "Mô hình chưa chắc chắn (dưới 60%). Hãy chụp lại ảnh rõ nét hơn, "
            "đủ sáng, lấy cận một lá và so sánh với các dự đoán còn lại."
        )

    st.warning(
        "Mô hình luôn chọn một trong 6 nhãn đã học, kể cả khi ảnh không phải lá lúa. "
        "Kết quả chỉ mang tính tham khảo; hãy kiểm tra thực tế trước khi xử lý cây trồng."
    )

    if predicted_class != "Healthy Rice Leaf":
        st.subheader("Ước lượng mức độ nặng")
        severity = estimate_severity(resized_image)
        if severity is None:
            st.info("Không tìm thấy đủ vùng lá trong ảnh để ước lượng mức độ nặng.")
        else:
            percent, level, severity_overlay = severity
            col1, col2 = st.columns(2)
            col1.metric("Diện tích lá có dấu hiệu bệnh", f"{percent:.1f}%")
            col2.metric("Mức độ", level)
            st.image(
                severity_overlay,
                caption="Vùng màu đỏ là phần được ước lượng là bị bệnh",
                width="stretch",
            )
            st.caption(SEVERITY_NOTE)

    info = DISEASE_INFO[predicted_class]
    st.subheader(f"Gợi ý phòng và trị: {DISPLAY_NAMES[predicted_class]}")
    st.markdown(f"**Triệu chứng:** {info['symptoms']}")
    st.markdown(f"**Nguyên nhân:** {info['cause']}")
    st.markdown("**Biện pháp phòng bệnh:**")
    for item in info["prevention"]:
        st.markdown(f"- {item}")
    st.markdown(f"**Hướng xử lý:** {info['treatment']}")
    st.caption(
        "Thông tin mang tính tham khảo chung, không nêu liều lượng. "
        "Hãy đọc kỹ nhãn thuốc và hỏi cán bộ khuyến nông hoặc bảo vệ thực vật địa phương trước khi phun."
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
