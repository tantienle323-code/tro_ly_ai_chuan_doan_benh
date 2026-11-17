from pathlib import Path
import os

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
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
    st.session_state["last_diagnosis"] = {
        "disease": DISPLAY_NAMES[predicted_class],
        "confidence": confidence,
        "top3": [
            (DISPLAY_NAMES[CLASS_NAMES[i]], float(predictions[i])) for i in top3_idx
        ],
    }

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


# ====================== CHATBOT HỎI ĐÁP BỆNH LÚA (GEMINI) ======================
CHAT_MODEL = "gemini-3.5-flash"
MAX_HISTORY = 10

BASE_SYSTEM_PROMPT = """Bạn là trợ lý tư vấn bệnh trên cây lúa, trả lời bằng tiếng Việt, dễ hiểu cho nông dân.
Quy tắc:
- Chỉ trả lời về lúa: bệnh, sâu hại, dinh dưỡng, canh tác, phòng trừ. Câu hỏi ngoài chủ đề thì lịch sự từ chối và mời hỏi về lúa.
- Không đưa liều lượng, nồng độ hay lịch phun thuốc cụ thể. Có thể nêu nhóm hoạt chất phổ biến, rồi nhắc đọc nhãn thuốc và hỏi cán bộ khuyến nông hoặc bảo vệ thực vật địa phương.
- Ưu tiên biện pháp canh tác và phòng bệnh trước khi dùng thuốc.
- Nếu không chắc chắn, hãy nói rõ là không chắc, không bịa thông tin.
- Kết quả chẩn đoán bằng AI chỉ là tham khảo, có thể sai; khuyên người dùng kiểm tra thực tế ngoài ruộng khi cần.
Cách trình bày:
- Trả lời gọn, tối đa khoảng 400 từ, đi thẳng vào ý chính.
- Không dùng tiêu đề lớn (#, ##). Chỉ dùng gạch đầu dòng ngắn và chữ in đậm cho ý quan trọng.
- Luôn kết thúc bằng một câu nhắc hỏi cán bộ khuyến nông địa phương khi cần dùng thuốc.
- Nếu người dùng muốn chi tiết hơn, họ sẽ hỏi thêm."""


def get_api_key():
    try:
        return st.secrets["GEMINI_API_KEY"]
    except (KeyError, FileNotFoundError):
        return os.environ.get("GEMINI_API_KEY")


def build_system_prompt():
    prompt = BASE_SYSTEM_PROMPT
    diagnosis = st.session_state.get("last_diagnosis")
    if diagnosis:
        top3_text = ", ".join(f"{name} ({prob:.0%})" for name, prob in diagnosis["top3"])
        prompt += (
            "\n\nNgữ cảnh: người dùng vừa tải lên một ảnh lá lúa. "
            f"Mô hình AI dự đoán: {diagnosis['disease']} (độ tin cậy {diagnosis['confidence']:.0%}). "
            f"Top 3 dự đoán: {top3_text}. Đây chỉ là dự đoán tham khảo."
        )
    return prompt


def to_gemini_contents(messages):
    messages = list(messages)
    while messages and messages[0]["role"] != "user":
        messages.pop(0)  # lịch sử phải bắt đầu bằng tin nhắn người dùng
    return [
        types.Content(
            role="user" if m["role"] == "user" else "model",
            parts=[types.Part(text=m["content"])],
        )
        for m in messages
    ]


def stream_gemini_reply(client, contents):
    stream = client.models.generate_content_stream(
        model=CHAT_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=build_system_prompt(),
            max_output_tokens=3000,
        ),
    )
    for chunk in stream:
        if chunk.text:
            st.session_state["partial_reply"] += chunk.text
            yield chunk.text


st.divider()
st.subheader("Hỏi đáp về bệnh lúa")

api_key = get_api_key()
if not api_key:
    st.info(
        "Chưa cấu hình GEMINI_API_KEY nên chatbot đang tắt. "
        "Hãy thêm key vào .streamlit/secrets.toml rồi chạy lại app."
    )
else:
    ss = st.session_state
    ss.setdefault("chat_messages", [])
    ss.setdefault("phase", "idle")  # idle | queued | running
    ss.setdefault("partial_reply", "")
    ss.setdefault("q_input", "")

    if ss["phase"] == "running":
        partial = ss["partial_reply"].strip()
        stopped_note = "*(đã dừng trả lời)*"
        ss["chat_messages"].append(
            {
                "role": "assistant",
                "content": f"{partial}\n\n{stopped_note}" if partial else stopped_note,
            }
        )
        ss["phase"] = "idle"
        ss["partial_reply"] = ""

    def submit_question():
        text = ss["q_input"].strip()
        if text and ss["phase"] == "idle":
            ss["chat_messages"].append({"role": "user", "content": text})
            ss["phase"] = "queued"
            ss["q_input"] = ""

    busy = ss["phase"] != "idle"

    chat_area = st.container()
    with chat_area:
        for message in ss["chat_messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    col_input, col_send, col_stop = st.columns([6, 1.3, 1.8], vertical_alignment="bottom")
    col_input.text_input(
        "Câu hỏi",
        key="q_input",
        placeholder="Hỏi về bệnh lúa, cách phòng trị, chăm sóc...",
        label_visibility="collapsed",
        disabled=busy,
        on_change=submit_question,
    )
    col_send.button("Gửi", key="send_btn", on_click=submit_question, disabled=busy, width="stretch")
    col_stop.button("⏹ Dừng", key="stop_btn", disabled=not busy, type="primary", width="stretch")  # chữ trên nút

    if ss["chat_messages"] and st.button("Xóa cuộc trò chuyện", disabled=busy):
        ss["chat_messages"] = []
        st.rerun()

    if ss["phase"] == "queued":
        ss["phase"] = "running"
        ss["partial_reply"] = ""
        history = ss["chat_messages"][-MAX_HISTORY:]

        with chat_area:
            with st.chat_message("assistant"):
                try:
                    client = genai.Client(api_key=api_key)
                    reply = st.write_stream(
                        stream_gemini_reply(client, to_gemini_contents(history))
                    )
                    if not reply:
                        reply = "Chatbot không trả lời được câu này, bạn thử diễn đạt lại nhé."
                        st.warning(reply)
                except genai_errors.APIError as exc:
                    if exc.code in (400, 401, 403):
                        reply = "API key không hợp lệ hoặc chưa được cấp quyền. Hãy kiểm tra lại secrets.toml."
                    elif exc.code == 429:
                        reply = "Đang gửi quá nhiều yêu cầu hoặc hết hạn mức miễn phí, hãy thử lại sau ít phút."
                    else:
                        reply = f"Không kết nối được chatbot (mã lỗi {exc.code}). Hãy thử lại sau."
                    st.error(reply)

        ss["chat_messages"].append({"role": "assistant", "content": reply})
        ss["phase"] = "idle"
        ss["partial_reply"] = ""
        st.rerun()