
import streamlit as st
from weather_service import search_location, get_weather

st.set_page_config(
    page_title="Thời tiết ruộng lúa",
    page_icon="🌾",
    layout="centered"
)

st.title("🌾 TRỢ LÝ AI NÔNG NGHIỆP")
st.subheader("Dự báo thời tiết tại ruộng lúa")
st.write("Chọn tỉnh/thành phố và khu vực gần ruộng lúa của bạn.")

# Danh sách 34 tỉnh/thành phố hiện hành
PROVINCES = {
    "An Giang": "An Giang",
    "Bắc Ninh": "Bac Ninh",
    "Cao Bằng": "Cao Bang",
    "Cà Mau": "Ca Mau",
    "Cần Thơ": "Can Tho",
    "Đà Nẵng": "Da Nang",
    "Đắk Lắk": "Dak Lak",
    "Điện Biên": "Dien Bien",
    "Đồng Nai": "Dong Nai",
    "Đồng Tháp": "Dong Thap",
    "Gia Lai": "Gia Lai",
    "Hà Nội": "Ha Noi",
    "Hà Tĩnh": "Ha Tinh",
    "Hải Phòng": "Hai Phong",
    "Huế": "Hue",
    "Hưng Yên": "Hung Yen",
    "Khánh Hòa": "Khanh Hoa",
    "Lai Châu": "Lai Chau",
    "Lâm Đồng": "Lam Dong",
    "Lạng Sơn": "Lang Son",
    "Lào Cai": "Lao Cai",
    "Nghệ An": "Nghe An",
    "Ninh Bình": "Ninh Binh",
    "Phú Thọ": "Phu Tho",
    "Quảng Ngãi": "Quang Ngai",
    "Quảng Ninh": "Quang Ninh",
    "Quảng Trị": "Quang Tri",
    "Sơn La": "Son La",
    "Tây Ninh": "Tay Ninh",
    "Thái Nguyên": "Thai Nguyen",
    "Thanh Hóa": "Thanh Hoa",
    "Thành phố Hồ Chí Minh": "Ho Chi Minh City",
    "Tuyên Quang": "Tuyen Quang",
    "Vĩnh Long": "Vinh Long"
}

# Chọn tỉnh/thành phố
province = st.selectbox(
    "📍 Chọn tỉnh/thành phố:",
    options=list(PROVINCES.keys()),
    index=list(PROVINCES.keys()).index("Thành phố Hồ Chí Minh")
)

# Tìm các địa điểm thuộc tỉnh/thành phố đã chọn
if st.button("🔍 Tìm khu vực ruộng lúa", use_container_width=True):
    try:
        with st.spinner("Đang tìm địa điểm..."):
            results = search_location(PROVINCES[province])

        # Ưu tiên các kết quả thuộc tỉnh đã chọn
        matched = [
            loc for loc in results
            if (
                PROVINCES[province].lower()
                in str(loc.get("admin1", "")).lower()
                or province.lower()
                in str(loc.get("admin1", "")).lower()
                or PROVINCES[province].lower()
                in str(loc.get("name", "")).lower()
            )
        ]

        st.session_state["locations"] = matched or results
        st.session_state["selected_province"] = province

        if matched or results:
            st.success("Đã tìm thấy địa điểm!")
        else:
            st.warning(
                "Chưa tìm được địa điểm. "
                "Bạn thử chọn tỉnh/thành phố khác nhé."
            )

    except Exception as e:
        st.error(f"Lỗi tìm địa điểm: {e}")

# Hiển thị danh sách địa điểm tìm được
locations = st.session_state.get("locations", [])

if locations:
    location_options = {
        i: (
            f"{loc['name']}, "
            f"{loc.get('admin2', '')}, "
            f"{loc.get('admin1', '')}"
        )
        for i, loc in enumerate(locations)
    }

    selected = st.selectbox(
        "🌾 Chọn khu vực gần ruộng lúa:",
        options=list(location_options.keys()),
        format_func=lambda i: location_options[i]
    )

    st.caption(
        "Tọa độ địa điểm được dùng để lấy thời tiết "
        "khu vực gần ruộng lúa."
    )

    if st.button(
        "🌦️ Lấy dự báo thời tiết",
        use_container_width=True
    ):
        location = locations[selected]

        try:
            with st.spinner("Đang lấy dữ liệu thời tiết..."):
                weather = get_weather(
                    location["latitude"],
                    location["longitude"]
                )

            st.success("Lấy dữ liệu thời tiết thành công!")

            st.write(
                f"📍 Địa điểm: {location_options[selected]}"
            )

            daily = weather["daily"]
            hourly = weather["hourly"]

            rows = []

            for i, day in enumerate(daily["time"]):
                # Tính độ ẩm trung bình theo ngày
                humidity_values = [
                    value
                    for time, value in zip(
                        hourly["time"],
                        hourly["relative_humidity_2m"]
                    )
                    if time.startswith(day) and value is not None
                ]

                avg_humidity = (
                    round(
                        sum(humidity_values) / len(humidity_values),
                        1
                    )
                    if humidity_values else None
                )

                rows.append({
                    "Ngày": day,
                    "Nhiệt độ cao nhất (°C)": daily[
                        "temperature_2m_max"
                    ][i],
                    "Nhiệt độ thấp nhất (°C)": daily[
                        "temperature_2m_min"
                    ][i],
                    "Độ ẩm trung bình (%)": avg_humidity,
                    "Lượng mưa (mm)": daily[
                        "precipitation_sum"
                    ][i],
                    "Xác suất mưa (%)": daily[
                        "precipitation_probability_max"
                    ][i]
                })

            st.subheader("📊 Dự báo thời tiết 5 ngày")

            st.dataframe(
                rows,
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                "Nguồn dữ liệu: Open-Meteo. "
                "Thời tiết thực tế có thể thay đổi."
            )

        except Exception as e:
            st.error(f"Không thể lấy dữ liệu thời tiết: {e}")
