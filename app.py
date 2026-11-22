from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps, UnidentifiedImageError
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    decode_predictions,
    preprocess_input as mobilenet_preprocess,
)


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

RICE_MARKET_PRICES = {
    "Lúa OM 5451 (8.100 đ/kg)": 8100,
    "Lúa Đài Thơm 8 (8.400 đ/kg)": 8400,
    "Lúa ST24 / ST25 (9.500 đ/kg)": 9500,
    "Lúa IR 50404 (7.600 đ/kg)": 7600,
    "Tự nhập giá lúa tại địa phương": 8000,
}

# Các từ khóa vật thể lạ (chai thuốc, bao bì, đồ vật, động vật, người...) nếu Gatekeeper phát hiện
NON_PLANT_KEYWORDS = {
    "bottle", "pill_bottle", "water_bottle", "pop_bottle", "beer_bottle", "wine_bottle",
    "packet", "carton", "box", "can", "tin", "jar", "jug", "cup", "mug", "spray",
    "lotion", "sunscreen", "shampoo", "soap", "perfume", "medicine", "syringe",
    "phone", "cellular_telephone", "laptop", "computer", "keyboard", "mouse", "screen",
    "book", "paper", "envelope", "menu", "comic_book", "web_site",
    "person", "man", "woman", "hand", "finger", "face", "shoe", "shirt",
    "car", "bicycle", "motorcycle", "dog", "cat", "bird", "fish", "toy", "ball",
}

# Các loài thực vật/hoa quả khác không phải lá lúa
OTHER_PLANT_KEYWORDS = {
    "daisy", "rose", "sunflower", "tulip", "orchid", "lotus", "hibiscus",
    "banana", "pineapple", "orange", "lemon", "apple", "strawberry", "mango",
    "cabbage", "broccoli", "cauliflower", "cucumber", "pumpkin", "mushroom",
    "cactus", "pine", "oak", "maple", "fern", "moss",
}

DISEASE_ADVISORY_DB = {
    "Bacterial Leaf Blight": {
        "tac_nhan": "Vi khuẩn Xanthomonas oryzae (không phải nấm, dùng thuốc trừ nấm sẽ không khỏi)",
        "ty_le_mat_min": 0.15,
        "ty_le_mat_max": 0.25,
        "thuoc_dac_tri": {
            "Totan 200WP (Bronopol) - Phổ thông": 850000,
            "Starner 20WP (Oxolinic acid) - Đặc trị cao cấp": 1150000,
            "Kamsu 2SL (Kasugamycin) - Sinh học an toàn": 950000,
        },
        "thoi_tiet_thuan_loi": ["Mưa nhiều / Ẩm độ cao", "Nắng nóng xen kẽ mưa giông"],
        "co_che_lay_lan": (
            "Vi khuẩn tiết dịch nhờn trên mép lá vào sáng sớm; khi có gió mạnh, mưa giông hoặc lá cọ xát vào nhau, "
            "vi khuẩn xâm nhập qua vết thương cơ giới và lỗ khí khổng, lây lan cực nhanh theo hướng gió và dòng nước tưới."
        ),
        "dau_hieu_kiem_tra": "Ra ruộng vào sáng sớm lúc còn sương, soi mặt dưới mép lá bị vàng xem có giọt dịch vi khuẩn nhỏ màu vàng đục như keo hay không.",
        "quan_ly_canh_tac": (
            "1. Rút bớt nước trong ruộng (chỉ giữ ẩm mặt đất) để cắt đường lây lan của vi khuẩn qua nước.\n"
            "2. **Ngưng bón phân đạm (Urê) và tuyệt đối không phun phân bón lá, thuốc kích thích sinh trưởng** (càng phun lá càng non mềm, bệnh bùng càng mạnh)."
        ),
        "ky_thuat_phun": (
            "Phun khi lá lúa đã ráo sương (khoảng 8h–10h sáng hoặc 15h30–17h chiều). "
            "Không lội ruộng phun khi lá còn ướt sương vì người đi tới đâu sẽ làm vi khuẩn lây lan theo tới đó. "
            "Nếu ruộng đang bị nặng, phun lặp lại lần 2 sau 5–7 ngày và bổ sung thêm Kali trắng giúp cứng lá."
        ),
    },
    "Brown Spot": {
        "tac_nhan": "Nấm Bipolaris oryzae (thường xuất hiện khi ruộng thiếu dinh dưỡng, phèn hoặc khô hạn)",
        "ty_le_mat_min": 0.10,
        "ty_le_mat_max": 0.18,
        "thuoc_dac_tri": {
            "Tilt Super 300EC (Propiconazole + Difenoconazole)": 920000,
            "Dithane M-45 80WP (Mancozeb) - Tiết kiệm": 750000,
            "Amistar Top 325SC (Azoxystrobin) - Dưỡng lá xanh": 1100000,
        },
        "thoi_tiet_thuan_loi": ["Nắng nóng xen kẽ mưa giông", "Khô hạn / Thiếu dinh dưỡng"],
        "co_che_lay_lan": (
            "Bào tử nấm phát tán nhờ gió từ các lá già phía dưới lên các lá non phía trên. "
            "Nếu ruộng bị thiếu nước hoặc đất phèn/nghèo Kali, cây suy yếu khiến vết đốm nâu nối liền nhau làm khô cháy cả lá."
        ),
        "dau_hieu_kiem_tra": "Kiểm tra các lá già tầng dưới xem có đốm tròn/bầu dục màu nâu sẫm giống hạt mè, xung quanh có quầng vàng nhạt hay không.",
        "quan_ly_canh_tac": (
            "1. Kiểm tra mực nước ruộng: không để ruộng bị khô nứt nẻ; nếu đất nhiễm phèn cần thay nước mới.\n"
            "2. Bón bổ sung phân **Kali và Silic** (hoặc phân NPK cân đối) để tăng độ dày vách tế bào lá, giúp cây tự kháng nấm."
        ),
        "ky_thuat_phun": (
            "Phun ướt đều cả mặt trên và mặt dưới tán lá. Có thể pha chung thuốc trừ nấm với phân bón lá chứa Silic hoặc Kali hữu cơ "
            "để vừa diệt nấm vừa phục hồi màu xanh cho bộ lá đòng."
        ),
    },
    "Leaf Blast": {
        "tac_nhan": "Nấm Pyricularia oryzae (bệnh nguy hiểm nhất trên lúa, có thể gây cháy rụi từng chòm)",
        "ty_le_mat_min": 0.15,
        "ty_le_mat_max": 0.30,
        "thuoc_dac_tri": {
            "Beam 75WP (Tricyclazole) - Đặc trị đạo ôn": 1150000,
            "Fuji-One 40EC (Isoprothiolane) - Hiệu quả nhanh": 1050000,
            "Filia 525SE (Tricyclazole + Propiconazole) - Cao cấp": 1350000,
        },
        "thoi_tiet_thuan_loi": ["Mưa nhiều / Ẩm độ cao", "Trời âm u / Sương mù nhiều"],
        "co_che_lay_lan": (
            "Ban đêm trời mát (20–26°C) kèm sương mù dày hoặc mưa phùn là điều kiện lý tưởng để nấm sinh bào tử ồ ạt. "
            "Chỉ sau 3–4 ngày, các vết bệnh hình mắt én nhỏ sẽ liên kết lại làm cháy khô cả chòm lúa ('sụp mặt') và dễ lây lên cổ bông về sau."
        ),
        "dau_hieu_kiem_tra": "Vạch các khóm lúa xanh tốt (đặc biệt chỗ rậm rạp, thừa đạm hoặc ven bờ) tìm vết chấm kim màu xám xanh hoặc vết hình thoi (mắt én) tâm xám viền nâu.",
        "quan_ly_canh_tac": (
            "1. **Giữ mực nước trong ruộng ổn định (3–5 cm)** — tuyệt đối không tháo cạn nước vì ruộng khô làm đạo ôn bùng phát dữ dội hơn.\n"
            "2. **Ngưng ngay phân đạm (Urê) và phân bón lá** cho đến khi vết bệnh khô hẳn (tâm vết bệnh chuyển sang màu xám trắng khô ráo)."
        ),
        "ky_thuat_phun": (
            "Phun thuốc ngay khi phát hiện vết chấm kim hoặc vết mắt én đầu tiên; phun kỹ vào các chòm lúa xanh đậm, rậm rạp. "
            "Nếu ruộng đã chớm cháy chòm, bắt buộc **phun kép 2 lần cách nhau 5–7 ngày** (nên đổi hoạt chất ở lần 2 để tránh kháng thuốc)."
        ),
    },
    "Leaf scald": {
        "tac_nhan": "Nấm Microdochium oryzae (thường xuất hiện từ giai đoạn làm đòng đến trỗ)",
        "ty_le_mat_min": 0.10,
        "ty_le_mat_max": 0.20,
        "thuoc_dac_tri": {
            "Carbenzim 500FL (Carbendazim) - Phổ thông": 780000,
            "Amistar Top 325SC (Azoxystrobin) - Đặc trị": 1100000,
            "Nativo 750WG (Tebuconazole + Trifloxystrobin)": 1250000,
        },
        "thoi_tiet_thuan_loi": ["Mưa nhiều / Ẩm độ cao", "Trời âm u / Sương mù nhiều"],
        "co_che_lay_lan": (
            "Nấm tấn công từ chóp lá và mép lá rồi lan dần xuống dưới tạo thành các vân gợn sóng màu nâu xám. "
            "Mưa gió làm các lá cọ xát và bào tử nấm theo nước mưa lan sang các khóm bên cạnh, làm giảm diện tích quang hợp nuôi hạt."
        ),
        "dau_hieu_kiem_tra": "Quan sát chóp các lá tầng giữa và lá đòng xem có vết cháy khô màu nâu nhạt, có các đường vân ngang như hình bậc thang/gợn sóng hay không.",
        "quan_ly_canh_tac": (
            "1. Giữ mực nước ruộng vừa phải, tránh để ruộng ngập quá sâu hoặc khô hạn kéo dài.\n"
            "2. Hạn chế bón thừa đạm vào đợt đón đòng; tăng cường Kali để giữ bộ lá đòng xanh bền đến lúc thu hoạch."
        ),
        "ky_thuat_phun": (
            "Phun phủ đều tầng lá trên của ruộng lúa vào sáng sớm hoặc chiều mát. "
            "Nếu lúa đang chuẩn bị trỗ, việc phun thuốc nhóm Strobilurin (như Azoxystrobin, Trifloxystrobin) sẽ vừa chặn bệnh vừa giúp xanh lá đòng, sáng hạt."
        ),
    },
    "Sheath Blight": {
        "tac_nhan": "Nấm Rhizoctonia solani (tồn tại sẵn trong đất/rơm rạ dưới dạng hạch nấm)",
        "ty_le_mat_min": 0.12,
        "ty_le_mat_max": 0.25,
        "thuoc_dac_tri": {
            "Validacin 5SL (Validamycin) - Tiết kiệm hiệu quả": 820000,
            "Anvil 5SC (Hexaconazole) - Chặn đứng vết bệnh": 980000,
            "Nevo 330EC (Propiconazole + Difenoconazole)": 1180000,
        },
        "thoi_tiet_thuan_loi": ["Mưa nhiều / Ẩm độ cao", "Nắng nóng xen kẽ mưa giông"],
        "co_che_lay_lan": (
            "Hạch nấm nổi trên mặt nước bám vào bẹ lúa sát gốc rồi bò dần lên trên theo chiều cao cây. "
            "Ruộng sạ dày, bón thừa đạm, tán lá rậm rạp nóng ẩm khiến nấm leo thẳng lên lá đòng chỉ trong 4–5 ngày, làm lúa lép lửng và đổ ngã."
        ),
        "dau_hieu_kiem_tra": "Vạch sát gốc lúa gần mặt nước xem bẹ lá có các vết đốm loang lổ hình bầu dục màu xám xanh như vết vằn da hổ và có hạt hạch nấm nhỏ màu nâu bám vào không.",
        "quan_ly_canh_tac": (
            "1. **Tháo cạn nước ruộng phơi mặt đất 3–5 ngày** để gốc lúa khô ráo, thông thoáng — đây là cách cắt đứt đường leo của nấm hiệu quả nhất.\n"
            "2. Ngưng bón đạm; nếu ruộng quá rậm có thể tỉa bớt lá chân ở bờ."
        ),
        "ky_thuat_phun": (
            "**Lưu ý quan trọng khi phun:** Không phun lướt trên ngọn lá! Phải chỉnh béc phun sương mịn và **hướng vòi phun xuống thấp vào vùng gốc và bẹ lúa** "
            "(tăng lượng nước phun để thuốc chảy dọc theo bẹ xuống sát gốc). Nếu bệnh đã leo qua nửa thân cây, cần phun nhắc lại sau 7 ngày."
        ),
    },
}


@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    if model.input_shape != (None, IMG_SIZE, IMG_SIZE, 3):
        raise ValueError(f"Kích thước đầu vào không đúng: {model.input_shape}")
    if model.output_shape != (None, len(CLASS_NAMES)):
        raise ValueError(f"Số lớp đầu ra không đúng: {model.output_shape}")
    return model


@st.cache_resource
def load_gatekeeper_model():
    """Tải mô hình MobileNetV2 (ImageNet) để phát hiện chai thuốc, vật dụng, con người, cây khác."""
    return MobileNetV2(weights="imagenet")


def kiem_tra_la_lua_nghiem_ngat(pil_image: Image.Image, gatekeeper_model):
    """
    Hệ thống lọc 3 lớp để chỉ chấp nhận ảnh LÁ LÚA:
    - Lớp 1: Dùng MobileNetV2 phát hiện vật thể lạ (chai thuốc, bao bì, đồ vật, con người, hoa quả...).
    - Lớp 2: Kiểm tra sắc tố thực vật đặc trưng của lá lúa trong không gian màu HSV.
    - Lớp 3: Kiểm tra cấu trúc biên cạnh & độ thẳng gân lá song song (đặc trưng cây họ lúa).
    """
    resized = pil_image.resize((224, 224))
    img_rgb = np.asarray(resized, dtype=np.float32)

    # --- LỚP 1: Gatekeeper phát hiện đồ vật / chai thuốc / sinh vật khác ---
    gk_input = mobilenet_preprocess(np.expand_dims(img_rgb.copy(), axis=0))
    gk_preds = gatekeeper_model.predict(gk_input, verbose=0)
    top_objects = decode_predictions(gk_preds, top=5)[0]

    for _, label, prob in top_objects:
        label_lower = label.lower()
        # Nếu mô hình tổng quát nhận ra chai thuốc, đồ vật, người... với xác suất > 12% -> chặn ngay
        if prob > 0.12 and any(kw in label_lower for kw in NON_PLANT_KEYWORDS):
            return (
                False,
                f"Không nhận dạng được do đây không phải lá lúa (hệ thống phát hiện vật thể giống **{label}**). "
                "Vui lòng chỉ tải lên ảnh chụp lá lúa.",
            )
        # Nếu nhận ra hoa, trái cây hoặc rau củ khác với xác suất > 20% -> chặn ngay
        if prob > 0.20 and any(kw in label_lower for kw in OTHER_PLANT_KEYWORDS):
            return (
                False,
                f"Không nhận dạng được do đây là loại cây/hoa quả khác (**{label}**), không phải lá lúa.",
            )

    # --- LỚP 2: Kiểm tra phân bố màu sắc đặc trưng của lá lúa ---
    img_uint8 = np.asarray(resized, dtype=np.uint8)
    img_hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)

    # Màu xanh diệp lục của lá lúa (Hue: 26 - 92)
    mask_green = cv2.inRange(img_hsv, np.array([26, 35, 25]), np.array([92, 255, 255]))
    # Màu vàng úa / nâu cháy của lá lúa bệnh (Hue: 10 - 26)
    mask_yellow_brown = cv2.inRange(img_hsv, np.array([10, 40, 35]), np.array([26, 255, 255]))
    # Màu trắng xóa / xám nền studio (thường gặp ở ảnh chụp chai thuốc trên nền trắng)
    mask_white_bg = cv2.inRange(img_hsv, np.array([0, 0, 210]), np.array([180, 35, 255]))

    ty_le_xanh = np.count_nonzero(mask_green) / mask_green.size
    ty_le_vang_nau = np.count_nonzero(mask_yellow_brown) / mask_yellow_brown.size
    ty_le_nen_trang = np.count_nonzero(mask_white_bg) / mask_white_bg.size

    if ty_le_nen_trang > 0.45 and ty_le_xanh < 0.25:
        return (
            False,
            "Không nhận dạng được do đây không phải lá lúa (ảnh có quá nhiều nền trắng giống ảnh sản phẩm/chai thuốc).",
        )

    if ty_le_xanh < 0.14 or (ty_le_xanh + ty_le_vang_nau) < 0.28:
        return (
            False,
            "Không nhận dạng được do đây không phải lá lúa (không tìm thấy đặc trưng màu sắc của lá lúa).",
        )

    # --- LỚP 3: Phân tích độ định hướng cấu trúc gân lá song song (Gradient Orientation) ---
    # Lá lúa thon dài, các mép lá và gân lá chạy song song theo một hướng ưu thế rõ rệt,
    # khác với lá cây thân gỗ bản rộng (gân tỏa tròn đa hướng) hoặc chữ in trên nhãn chai.
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.hypot(sobelx, sobely)
    angle = np.rad2deg(np.arctan2(sobely, sobelx)) % 180  # 0 - 180 độ

    # Chỉ xét các pixel thuộc vùng lá cây có biên cạnh rõ
    leaf_mask = cv2.bitwise_or(mask_green, mask_yellow_brown) > 0
    strong_edges = (magnitude > np.percentile(magnitude, 70)) & leaf_mask

    if np.count_nonzero(strong_edges) < 150:
        return (
            False,
            "Không nhận dạng được do ảnh mờ hoặc không thấy rõ phiến lá lúa.",
        )

    hist, _ = np.histogram(angle[strong_edges], bins=18, range=(0, 180), weights=magnitude[strong_edges])
    hist_norm = hist / (np.sum(hist) + 1e-8)
    # Tỷ lệ tập trung của 3 hướng góc liên tiếp lớn nhất (độ song song của lá lúa)
    padded_hist = np.concatenate([hist_norm, hist_norm[:2]])
    max_directional_ratio = max(np.sum(padded_hist[i : i + 3]) for i in range(18))

    # Nếu các đường biên phân tán lộn xộn đa hướng (< 0.26) -> lá cây bản rộng khác hoặc chữ trên bao bì
    if max_directional_ratio < 0.26:
        return (
            False,
            "Không nhận dạng được do hình dáng/gân lá không phải lá lúa (có thể là lá cây khác hoặc vật thể lạ).",
        )

    return True, "Hợp lệ"


def phan_tich_kinh_te_va_lay_lan(
    predicted_class: str,
    thoi_tiet: str,
    giai_doan_lua: str,
    dien_tich_ha: float,
    gia_lua_kg: int,
    don_gia_thuoc_ha: int,
    nang_suat_tan_ha: float = 6.5,
):
    if predicted_class == "Healthy Rice Leaf" or predicted_class not in DISEASE_ADVISORY_DB:
        return None

    info = DISEASE_ADVISORY_DB[predicted_class]
    tong_gia_tri_1ha = nang_suat_tan_ha * 1000 * gia_lua_kg

    ty_le_tb = (info["ty_le_mat_min"] + info["ty_le_mat_max"]) / 2
    thiet_hai_1ha = tong_gia_tri_1ha * ty_le_tb
    thiet_hai_ruong = thiet_hai_1ha * dien_tich_ha
    chi_phi_thuoc_ruong = don_gia_thuoc_ha * dien_tich_ha
    tien_cuu_duoc = thiet_hai_ruong - chi_phi_thuoc_ruong

    lit_nuoc_can = round(450 * dien_tich_ha)
    so_binh_25l = max(1, round(lit_nuoc_can / 25))

    diem_nguy_co = 1
    if thoi_tiet in info["thoi_tiet_thuan_loi"]:
        diem_nguy_co += 2
    if giai_doan_lua in ["Đẻ nhánh", "Làm đòng"]:
        diem_nguy_co += 1

    if diem_nguy_co >= 3:
        muc_do = "RẤT CAO (Nguy cơ bùng phát dịch)"
        mau_nguy_co = "#d32f2f"
        nen_nguy_co = "#ffebee"
        du_bao_3_5_ngay = (
            f"**Ngày 1–2:** Vết bệnh mới xuất hiện dày đặc trên các khóm lúa lân cận do {thoi_tiet.lower()} "
            f"kết hợp giai đoạn **{giai_doan_lua.lower()}** lá non nhiều.\n"
            f"- **Ngày 3–5:** Nếu không chặn đứng, tỷ lệ nhiễm có thể vượt quá **40–60% diện tích ruộng**, "
            f"gây cháy chòm hoặc làm giảm mạnh khả năng nuôi đòng/vô gạo."
        )
        thoi_diem_phun = "Phun xử lý NGAY TRONG 24–48 GIỜ TỚI (chọn lúc trời tạnh ráo, không mưa trong vòng 4 tiếng sau phun)."
    elif diem_nguy_co == 2:
        muc_do = "TRUNG BÌNH (Lây lan cục bộ)"
        mau_nguy_co = "#ef6c00"
        nen_nguy_co = "#fff3e0"
        du_bao_3_5_ngay = (
            "**Ngày 1–2:** Bệnh chủ yếu phát triển tại các ổ bệnh cũ và các chỗ sạ dày, trũng nước.\n"
            "- **Ngày 3–5:** Vết bệnh mở rộng kích thước và lan sang các khóm xung quanh trong bán kính 1–2m nếu gặp sương đêm hoặc mưa rào."
        )
        thoi_diem_phun = "Nên phun khoanh vùng dập dịch trong **2–3 ngày tới** để tránh phải phun lại nhiều lần."
    else:
        muc_do = "THẤP (Tiến triển chậm)"
        mau_nguy_co = "#2e7d32"
        nen_nguy_co = "#e8f5e9"
        du_bao_3_5_ngay = (
            "**Ngày 1–2:** Thời tiết hiện tại ức chế sự phát tán của mầm bệnh, vết bệnh cũ có xu hướng khô viền.\n"
            "- **Ngày 3–5:** Mầm bệnh vẫn còn tiềm ẩn trên đồng; chỉ bùng phát trở lại nếu thời tiết chuyển mưa ẩm hoặc bón thừa đạm."
        )
        thoi_diem_phun = "Chưa cần phun ồ ạt nếu tỷ lệ bệnh dưới 5% số lá; ưu tiên điều chỉnh nước, phân bón và theo dõi sát."

    return {
        "tac_nhan": info["tac_nhan"],
        "ty_le_mat": f"{int(info['ty_le_mat_min'] * 100)}–{int(info['ty_le_mat_max'] * 100)}%",
        "thiet_hai_1ha": thiet_hai_1ha,
        "thiet_hai_ruong": thiet_hai_ruong,
        "chi_phi_thuoc_ruong": chi_phi_thuoc_ruong,
        "tien_cuu_duoc": tien_cuu_duoc,
        "lit_nuoc_can": lit_nuoc_can,
        "so_binh_25l": so_binh_25l,
        "muc_do": muc_do,
        "mau_nguy_co": mau_nguy_co,
        "nen_nguy_co": nen_nguy_co,
        "co_che_lay_lan": info["co_che_lay_lan"],
        "du_bao_3_5_ngay": du_bao_3_5_ngay,
        "thoi_diem_phun": thoi_diem_phun,
        "dau_hieu_kiem_tra": info["dau_hieu_kiem_tra"],
        "quan_ly_canh_tac": info["quan_ly_canh_tac"],
        "ky_thuat_phun": info["ky_thuat_phun"],
    }


st.set_page_config(
    page_title="Nhận diện bệnh qua ảnh lá lúa",
    page_icon="🌾",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #f4f7f5;
    }
    .hero-banner {
        background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 60%, #43a047 100%);
        padding: 1.8rem 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 6px 18px rgba(27, 94, 32, 0.18);
    }
    .hero-banner h1 {
        color: white !important;
        margin: 0 0 0.4rem 0;
        font-size: 1.9rem;
        font-weight: 700;
    }
    .hero-banner p {
        color: #e8f5e9;
        margin: 0;
        font-size: 1.02rem;
    }
    .card-box {
        background-color: white;
        padding: 1.3rem 1.5rem;
        border-radius: 14px;
        border: 1px solid #e0e8e3;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.04);
        margin-bottom: 1rem;
    }
    .diagnosis-badge {
        background: #e8f5e9;
        border-left: 6px solid #2e7d32;
        padding: 1rem 1.2rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .diagnosis-badge-sick {
        background: #fff8e1;
        border-left: 6px solid #f57c00;
        padding: 1rem 1.2rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        padding: 0.85rem 1rem;
        border-radius: 12px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.03);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-banner">
        <h1>🌾 Nhận diện bệnh qua ảnh lá lúa</h1>
        <p>Tính toán thiệt hại thực tế • Dự báo lây lan 3–5 ngày • Phác đồ xử lý ngoài đồng</p>
    </div>
    """,
    unsafe_allow_html=True,
)


if not MODEL_PATH.is_file():
    st.error(f"Không tìm thấy model: {MODEL_PATH.name}. Hãy đặt model cạnh app.py.")
    st.stop()

try:
    model = load_model()
    gatekeeper_model = load_gatekeeper_model()
except (OSError, ValueError) as exc:
    st.error(f"Không thể nạp model: {exc}")
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


left_col, right_col = st.columns([1.05, 1.35], gap="large")

with left_col:
    with st.container(border=True):
        st.subheader("1. Thông tin đồng ruộng")

        c_dt1, c_dt2 = st.columns([1.1, 1])
        with c_dt1:
            don_vi_dt = st.selectbox("Đơn vị đo", ["Hecta (ha)", "Công tầm lớn (1.300m²)"])
        with c_dt2:
            so_luong_dt = st.number_input("Diện tích ruộng", min_value=0.1, value=1.0, step=0.5)

        dien_tich_ha = so_luong_dt if don_vi_dt == "Hecta (ha)" else round(so_luong_dt * 0.13, 2)
        if don_vi_dt != "Hecta (ha)":
            st.caption(f"Quy đổi tương đương: **{dien_tich_ha} ha**")

        c_gia1, c_gia2 = st.columns([1.4, 1])
        with c_gia1:
            giong_lua = st.selectbox("Giống lúa (tự lấy giá)", list(RICE_MARKET_PRICES.keys()))
        with c_gia2:
            gia_lua_hien_tai = st.number_input(
                "Giá lúa (đ/kg)",
                min_value=4000,
                max_value=20000,
                value=RICE_MARKET_PRICES[giong_lua],
                step=100,
            )

        c_dk1, c_dk2 = st.columns(2)
        with c_dk1:
            thoi_tiet = st.selectbox(
                "Thời tiết hiện tại",
                [
                    "Mưa nhiều / Ẩm độ cao",
                    "Nắng nóng xen kẽ mưa giông",
                    "Trời âm u / Sương mù nhiều",
                    "Khô hạn / Thiếu dinh dưỡng",
                ],
            )
        with c_dk2:
            giai_doan = st.selectbox(
                "Giai đoạn lúa",
                ["Mạ", "Đẻ nhánh", "Làm đòng", "Trỗ – Chín"],
                index=2,
            )

    with st.container(border=True):
        st.subheader("2. Tải ảnh lá lúa cần kiểm tra")
        uploaded_file = st.file_uploader(
            "Chỉ tải lên ảnh chụp lá lúa (JPG, PNG)",
            type=["jpg", "jpeg", "png"],
            max_upload_size=10,
        )

with right_col:
    if uploaded_file is None:
        st.markdown(
            """
            <div class="card-box" style="text-align: center; padding: 3rem 1.5rem;">
                <h3 style="color: #2e7d32;">👈 Sẵn sàng phân tích</h3>
                <p style="color: #555;">Hãy nhập diện tích ruộng và tải ảnh lá lúa ở cột bên trái để AI chẩn đoán bệnh, tính toán chi phí và lập phác đồ xử lý.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
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

        # Kiểm tra nghiêm ngặt: chỉ cho phép ảnh LÁ LÚA đi tiếp
        la_lua_hop_le, ly_do_tu_choi = kiem_tra_la_lua_nghiem_ngat(image, gatekeeper_model)

        if not la_lua_hop_le:
            st.error(f"🚫 **{ly_do_tu_choi}**")
            st.image(image, caption="Ảnh bị từ chối nhận dạng", width=340)
            st.info(
                "📌 **Hệ thống chỉ hỗ trợ nhận dạng trên LÁ LÚA.**\n"
                "Vui lòng tải lên ảnh chụp cận cảnh 1–2 phiến lá lúa rõ nét (không chụp chai thuốc, đồ vật hay lá cây khác)."
            )
            st.stop()

        resized_image = image.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.expand_dims(
            np.asarray(resized_image, dtype=np.float32), axis=0
        ) / 255.0
        predictions = model.predict(img_array, verbose=0)[0]

        if predictions.shape != (len(CLASS_NAMES),) or not np.isfinite(predictions).all():
            st.error("Model trả về kết quả không hợp lệ. Hãy kiểm tra lại file model.")
            st.stop()

        top6_indices = np.argsort(predictions)[::-1][:6]
        top1_index = int(top6_indices[0])
        top1_class = CLASS_NAMES[top1_index]
        top1_confidence = float(predictions[top1_index])

        badge_class = "diagnosis-badge" if top1_class == "Healthy Rice Leaf" else "diagnosis-badge-sick"
        st.markdown(
            f"""
            <div class="{badge_class}">
                <div style="font-size: 0.9rem; color: #555; font-weight: 600;">KẾT QUẢ NHẬN DIỆN CAO NHẤT (TOP 1)</div>
                <div style="font-size: 1.55rem; font-weight: 700; color: #1b5e20; margin: 0.2rem 0;">
                    {DISPLAY_NAMES[top1_class]}
                </div>
                <div style="font-size: 0.95rem; color: #333;">Độ tin cậy mô hình: <b>{top1_confidence:.1%}</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.container(border=True):
            st.markdown("**📊 Top 6 chẩn đoán có xác suất cao nhất từ AI:**")
            for rank, idx in enumerate(top6_indices, start=1):
                cls_name = CLASS_NAMES[int(idx)]
                prob = float(predictions[int(idx)])
                c_name, c_bar, c_pct = st.columns([2.2, 3.5, 1])
                with c_name:
                    label = f"**{rank}. {DISPLAY_NAMES[cls_name]}**" if rank == 1 else f"{rank}. {DISPLAY_NAMES[cls_name]}"
                    st.write(label)
                with c_bar:
                    st.progress(min(max(prob, 0.0), 1.0))
                with c_pct:
                    st.write(f"**{prob:.1%}**" if rank == 1 else f"{prob:.1%}")

            top6_options = [CLASS_NAMES[int(i)] for i in top6_indices]
            selected_class = st.selectbox(
                "🔍 Chọn chẩn đoán để xem vùng AI tập trung (mặc định chọn Top 1):",
                options=top6_options,
                format_func=lambda x: f"{DISPLAY_NAMES[x]} ({predictions[CLASS_NAMES.index(x)]:.1%})",
            )
            selected_index = CLASS_NAMES.index(selected_class)

        img_c1, img_c2 = st.columns(2)
        with img_c1:
            st.image(image, caption="Ảnh lá lúa tải lên", use_container_width=True)
        with img_c2:
            try:
                heatmap = make_gradcam_heatmap(img_array, model, selected_index)
                if np.isfinite(heatmap).all() and np.any(heatmap):
                    image_array = np.asarray(resized_image)
                    image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                    heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
                    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
                    overlay = cv2.addWeighted(image_bgr, 0.6, heatmap_colored, 0.4, 0)
                    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                    st.image(
                        overlay,
                        caption=f"Vùng AI tập trung cho: {DISPLAY_NAMES[selected_class]}",
                        use_container_width=True,
                    )
            except Exception:
                st.caption("Không tạo được ảnh Grad-CAM cho mẫu này.")

        if selected_class == "Healthy Rice Leaf":
            st.success(
                "**Lá lúa đang khỏe mạnh**, chưa ghi nhận dấu hiệu sâu bệnh gây hại.\n\n"
                "- **Khuyến nghị:** Tiếp tục thăm đồng 2–3 ngày/lần vào sáng sớm, giữ mực nước vừa phải và bón phân cân đối N-P-K."
            )
