import os
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# Nạp module tin tức Tầng 3
from src.backtest import prepare_walk_forward_backtest_data, run_backtest
from src.data_quality import assess_market_data
from src.forecast_tracking import summarize_forecast_history, update_forecast_history
from src.auto_refresh import (
    DEFAULT_REFRESH_INTERVAL_MINUTES,
    run_refresh_if_due,
)
from src.refresh_pipeline import refresh_market_data as run_market_refresh_pipeline
from src.project_summary import build_project_summary

try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:  # The dashboard still works; install requirements for automatic browser refresh.
    st_autorefresh = None
from src.atomic_files import atomic_output_path
from src.sjc_prices import quote_comparison, validate_sjc_quotes
from src.news_sentiment import (
    NewsFetchError,
    NewsTranslationError,
    get_latest_news_sentiment,
    translate_headline_to_vietnamese,
)

st.set_page_config(
    page_title="Gold Market Desk | AI Analytics",
    page_icon="◈",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'gold_data_processed.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'gold_model.pkl')
FORECAST_HISTORY_PATH = os.path.join(BASE_DIR, 'data', 'forecast_history.sqlite3')
SJC_DATA_PATH = os.path.join(BASE_DIR, 'data', 'sjc_prices.csv')

@st.cache_data(ttl=300)
def load_data():
    if not os.path.exists(DATA_PATH):
        return None
    df = pd.read_csv(DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    return df.sort_values('Date').reset_index(drop=True)

def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)


@st.cache_data(ttl=900)
def load_news(refresh_key):
    return get_latest_news_sentiment()


@st.cache_data(ttl=86400, max_entries=256, show_spinner=False)
def translate_news_title(title):
    return translate_headline_to_vietnamese(title)


def refresh_market_data():
    return run_market_refresh_pipeline(BASE_DIR)


def style_chart(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        font=dict(family="Inter, sans-serif", color="#B8C4D0", size=12),
        margin=dict(l=18, r=18, t=36, b=18),
        hoverlabel=dict(bgcolor="#18232D", bordercolor="#344451", font_color="#F4F7FA"),
    )
    fig.update_xaxes(gridcolor="rgba(148, 163, 184, 0.10)", zerolinecolor="rgba(148, 163, 184, 0.15)")
    fig.update_yaxes(gridcolor="rgba(148, 163, 184, 0.10)", zerolinecolor="rgba(148, 163, 184, 0.15)")
    return fig

# Refresh the browser periodically while the dashboard is open. A separate
# worker/OS scheduler can refresh data even when no browser is connected.
if st_autorefresh is not None:
    st_autorefresh(interval=15 * 60 * 1000, key="gold_market_page_refresh")

try:
    AUTO_REFRESH_INTERVAL_MINUTES = max(
        1,
        int(
            os.getenv(
                "GOLD_AUTO_REFRESH_MINUTES",
                str(DEFAULT_REFRESH_INTERVAL_MINUTES),
            )
        ),
    )
except ValueError:
    AUTO_REFRESH_INTERVAL_MINUTES = DEFAULT_REFRESH_INTERVAL_MINUTES
try:
    auto_refresh_status = run_refresh_if_due(
        BASE_DIR,
        refresh_market_data,
        interval_minutes=AUTO_REFRESH_INTERVAL_MINUTES,
    )
    if auto_refresh_status.get("changed"):
        st.cache_data.clear()
except (OSError, RuntimeError, TimeoutError, ValueError) as auto_refresh_error:
    auto_refresh_status = {
        "status": "error",
        "message": str(auto_refresh_error),
    }

df = load_data()
model_payload = load_model()

if df is None or df.empty:
    st.error("Chưa tìm thấy dữ liệu đã xử lý. Hãy tải dữ liệu thị trường để tạo bộ dữ liệu dashboard.")
    if st.button("Tải và chuẩn bị dữ liệu thị trường"):
        with st.spinner("Đang tải dữ liệu, tính chỉ báo và kiểm định lại mô hình..."):
            refresh_market_data()
        st.cache_data.clear()
        st.rerun()
    st.stop()

st.html("""
<style>
[data-testid="stHeader"] { background: rgba(9, 15, 21, 0.94); }
[data-testid="stSidebar"] {
    background: #0C141B;
    border-right: 1px solid #22303A;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 1.5rem; }
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #EEF3F7; letter-spacing: -0.02em; }
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label { color: #B6C3CE; }
.block-container { max-width: 1440px; padding-top: 3rem; padding-bottom: 3.5rem; }
h1, h2, h3 { color: #F1F5F8; letter-spacing: -0.025em; }
h2 { font-size: 1.3rem !important; margin-top: 1.5rem !important; }
[data-testid="stCaptionContainer"] { color: #94A4B1; }
[data-testid="stMarkdownContainer"] p { line-height: 1.65; }
[data-testid="stMetric"] {
    background: linear-gradient(145deg, #121D26 0%, #0F1921 100%);
    border: 1px solid #263743;
    border-radius: 14px;
    padding: 17px 19px;
    box-shadow: 0 5px 18px rgba(0, 0, 0, 0.12);
    transition: border-color 160ms ease, transform 160ms ease;
}
[data-testid="stMetric"]:hover { border-color: #52616B; transform: translateY(-1px); }
[data-testid="stMetricLabel"] {
    color: #AAB8C3;
    font-size: 0.76rem;
    font-weight: 600;
    letter-spacing: 0.035em;
}
[data-testid="stMetricValue"] {
    color: #F3F6F8;
    font-family: 'Cascadia Code', Consolas, monospace;
    font-size: clamp(1.25rem, 2vw, 1.85rem);
}
[data-testid="stMetricDelta"] { font-size: 0.78rem; font-weight: 500; }
[data-testid="stTabs"] [role="tablist"] {
    gap: 0.45rem;
    border-bottom: 1px solid #25343F;
}
[data-testid="stTabs"] button[role="tab"] {
    color: #AEBBC6;
    padding: 0.85rem 1.1rem;
    border-radius: 8px 8px 0 0;
    transition: background 150ms ease, color 150ms ease;
}
[data-testid="stTabs"] button[role="tab"]:hover { color: #F4F6F8; background: #131F28; }
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    color: #F4CC68;
    border-bottom-color: #F2C14E;
}
[data-testid="stAlert"] { border-radius: 12px; border: 1px solid #2A3A45; }
.stButton > button,
[data-testid="stDownloadButton"] > button {
    border-radius: 9px;
    border: 1px solid #34444F;
    transition: border-color 150ms ease, background 150ms ease;
}
.stButton > button:hover,
[data-testid="stDownloadButton"] > button:hover { border-color: #F2C14E; }
.market-header {
    position: relative;
    display: flex; justify-content: space-between; align-items: center; gap: 1rem;
    padding: 1.55rem 1.55rem; margin: .5rem 0 1.5rem;
    background: linear-gradient(112deg, #14212A 0%, #111A22 68%, #172019 100%);
    border: 1px solid #2B3A44; border-radius: 16px;
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.16);
}
.market-header::before {
    position: absolute;
    inset: 0 auto 0 0;
    width: 3px;
    border-radius: 16px 0 0 16px;
    background: linear-gradient(180deg, #F2C14E, #B58A2C);
    content: "";
}
.market-brand { display: flex; align-items: center; gap: 1rem; min-width: 0; flex: 1 1 auto; }
.market-brand > div:last-child { min-width: 0; }
.market-logo {
    display: grid; place-items: center; width: 42px; height: 42px;
    flex: 0 0 42px;
    border: 1px solid rgba(242, 193, 78, 0.24);
    border-radius: 13px; background: rgba(242, 193, 78, 0.1);
    color: #F2C14E; font-weight: 800; font-family: 'Cascadia Code', Consolas, monospace;
}
.market-kicker {
    color: #B7C4CE; text-transform: uppercase; letter-spacing: .12em;
    font-size: .72rem; font-weight: 700; line-height: 1.6;
    white-space: nowrap;
}
.market-title { color: #F4F7F9; font-weight: 700; font-size: 1.2rem; letter-spacing: -0.02em; margin-top: .16rem; white-space: nowrap; }
.market-subtitle { color: #AAB8C3; font-size: .78rem; margin-top: .22rem; }
.market-meta { color: #B8C4CD; text-align: right; font-size: .74rem; line-height: 1.8; flex: 0 1 auto; }
.market-live { color: #66D19E; font-family: 'Cascadia Code', Consolas, monospace; }
.market-stale { color: #F2C14E; font-family: 'Cascadia Code', Consolas, monospace; }
div[data-testid="stPlotlyChart"] {
    border: 1px solid #24343E;
    border-radius: 14px;
    background: #0E161E;
    padding: 0.4rem;
}
@media (max-width: 760px) {
    .market-header { align-items: flex-start; flex-direction: column; }
    .market-meta { text-align: left; }
    [data-testid="stMetric"] { padding: 14px 15px; }
}
</style>
""")

latest_row = df.iloc[-1]
data_health = assess_market_data(df)
data_status_label = {
    'ready': 'DATA READY',
    'stale': 'DỮ LIỆU CHẬM',
    'error': 'DỮ LIỆU CÓ LỖI',
}[data_health['status']]
data_status_class = 'market-live' if data_health['status'] == 'ready' else 'market-stale'
st.html(f"""
<div class="market-header">
  <div class="market-brand">
    <div class="market-logo">Au</div>
    <div>
      <div class="market-kicker">Gold market intelligence</div>
      <div class="market-title">Gold Market Desk</div>
      <div class="market-subtitle">XAU · AI signal · risk monitoring</div>
    </div>
  </div>
  <div class="market-meta">
    <div><span class="{data_status_class}">● {data_status_label}</span></div>
    <div>Phiên dữ liệu mới nhất &nbsp; <strong>{latest_row['Date']:%d/%m/%Y}</strong></div>
    <div>{data_health['row_count']:,} dòng · trễ {data_health['business_days_lag']} ngày làm việc</div>
  </div>
</div>
""")
if data_health['status'] == 'error':
    st.error(
        "Dữ liệu có vấn đề về cấu trúc hoặc giá trị. Hãy cập nhật/kiểm tra dữ liệu trước khi "
        "dựa vào tín hiệu."
    )
elif data_health['status'] == 'stale':
    st.warning("Dữ liệu đã trễ hơn 5 ngày làm việc. Hãy cập nhật trước khi dùng tín hiệu.")
elif data_health['calendar_age_days'] and data_health['calendar_age_days'] >= 3:
    st.info("Thị trường có thể nghỉ cuối tuần/ngày lễ; hãy đối chiếu ngày phiên mới nhất.")

# SIDEBAR
st.sidebar.markdown("### MARKET DESK")
st.sidebar.caption("GOLD · QUANT ANALYTICS")
st.sidebar.divider()
st.sidebar.subheader("Bảng điều khiển")
if st.sidebar.button("↻ Cập nhật dữ liệu thị trường", use_container_width=True):
    try:
        with st.spinner("Đang tải dữ liệu, kiểm tra thay đổi và cập nhật mô hình nếu có dữ liệu mới..."):
            manual_status = run_refresh_if_due(
                BASE_DIR,
                refresh_market_data,
                interval_minutes=AUTO_REFRESH_INTERVAL_MINUTES,
                force=True,
            )
    except (OSError, RuntimeError, TimeoutError, ValueError) as error:
        st.error(f"Không thể cập nhật thị trường: {error}")
    else:
        if manual_status.get("status") == "error":
            st.error(f"Cập nhật thất bại: {manual_status.get('message', 'Lỗi không rõ')}")
        else:
            st.cache_data.clear()
            st.rerun()

st.sidebar.caption(f"Phiên dữ liệu mới nhất: {latest_row['Date']:%d/%m/%Y}")
refresh_schedule = (
    "mỗi ngày"
    if AUTO_REFRESH_INTERVAL_MINUTES == DEFAULT_REFRESH_INTERVAL_MINUTES
    else f"mỗi {AUTO_REFRESH_INTERVAL_MINUTES} phút"
)
st.sidebar.caption(f"Tự cập nhật: {refresh_schedule} khi dashboard mở")
if auto_refresh_status.get("status") == "error":
    st.sidebar.warning(f"Lần cập nhật tự động gần nhất thất bại: {auto_refresh_status.get('message', '')}")
elif auto_refresh_status.get("last_success_utc"):
    st.sidebar.caption(f"Lần kiểm tra tự động thành công gần nhất (UTC): {auto_refresh_status['last_success_utc']}")
with st.sidebar.expander("Chất lượng và độ mới dữ liệu"):
    st.write(f"Trạng thái: **{data_status_label}**")
    st.write(f"Số bản ghi: {data_health['row_count']:,}")
    st.write(f"Trễ: {data_health['business_days_lag']} ngày làm việc")
    st.write(f"Ngày lịch từ phiên gần nhất: {data_health['calendar_age_days']}")
    if data_health['issues']:
        for issue in data_health['issues']:
            st.warning(issue)
    else:
        st.success("Không phát hiện ngày trùng, giá đóng cửa lỗi hoặc ngày tương lai.")
num_days = st.sidebar.slider(
    "Khoảng thời gian biểu đồ (phiên)",
    min_value=1,
    max_value=len(df),
    value=min(180, len(df)),
)

st.sidebar.subheader("Theo dõi rủi ro")
enable_layer1 = st.sidebar.checkbox("Cảnh báo biến động", value=True)
enable_layer2 = st.sidebar.checkbox("Điều chỉnh vị thế theo biến động", value=True)
enable_layer3 = st.sidebar.checkbox("Phân tích tin tức", value=True)

st.sidebar.subheader("Cảnh báo Telegram")
telegram_direction_filter = st.sidebar.selectbox(
    "Cho phép gửi hướng",
    ['Tất cả', 'Chỉ tín hiệu tăng', 'Chỉ tín hiệu không tăng'],
    key='telegram_direction_filter_v2',
)
telegram_min_confidence = st.sidebar.slider(
    "Điểm xác suất tối thiểu để gửi (%)",
    min_value=50,
    max_value=95,
    value=60,
    step=5,
    key='telegram_min_confidence',
)
st.sidebar.caption("Chỉ gửi khi bạn chủ động bấm nút; ứng dụng không tự gửi tin.")

st.sidebar.subheader("Chỉ báo trên biểu đồ")
show_sma = st.sidebar.checkbox("Đường trung bình SMA", value=True)
show_bb = st.sidebar.checkbox("Dải Bollinger", value=True)
show_daily_vol = st.sidebar.checkbox("Lợi suất hàng ngày", value=True)

df_view = df.tail(num_days).copy()
forecast_history = pd.DataFrame()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Thị trường & tín hiệu",
    "Hiệu suất chiến lược",
    "Chất lượng mô hình",
    "Giá vàng SJC",
    "Tổng quan đồ án",
])

# ================= TAB 1 =================
with tab1:
    st.caption("MARKET SNAPSHOT")
    col1, col2 = st.columns(2)
    
    prev_close = df.iloc[-2]['Close'] if len(df) > 1 else latest_row['Close']
    diff = latest_row['Close'] - prev_close
    pct = (diff / prev_close) * 100 if prev_close else 0.0
    
    col1.metric("Vàng · USD/oz", f"${latest_row['Close']:,.2f}", f"{diff:+.2f} ({pct:+.2f}%)")
    
    if 'USD_Index' in latest_row:
        dxy_ret = latest_row.get('USD_Index_Return', 0.0)*100
        col2.metric("USD Index", f"{latest_row['USD_Index']:.2f}", f"{dxy_ret:+.2f}%")

    col3, col4 = st.columns(2)
    if 'US10Y_Yield' in latest_row:
        col3.metric("Lợi suất 10Y", f"{latest_row['US10Y_Yield']:.2f}%")
        
    daily_range = latest_row.get('Daily_Range_USD', 0.0)
    col4.metric("Biên độ phiên", f"${daily_range:.2f}", f"{latest_row.get('Daily_Range_Pct', 0.0):.2f}%")

    if model_payload is not None and data_health['status'] != 'error':
        model = model_payload['model']
        feature_cols = model_payload['feature_cols']
        input_vector = df.iloc[[-1]][feature_cols]

        pred = model.predict(input_vector)[0]
        confidence = float(model.predict_proba(input_vector)[0][pred])
        proba = confidence * 100
        return_model = model_payload.get('return_model')
        predicted_return_pct = (
            float(return_model.predict(input_vector)[0])
            if return_model is not None else 0.0
        )
        interval_half_width_pct = float(
            model_payload.get('prediction_interval_pct', 0.0)
        )
        expected_close = latest_row['Close'] * (1 + predicted_return_pct / 100)
        lower_close = latest_row['Close'] * (
            1 + (predicted_return_pct - interval_half_width_pct) / 100
        )
        upper_close = latest_row['Close'] * (
            1 + (predicted_return_pct + interval_half_width_pct) / 100
        )
        forecast_history = update_forecast_history(
            FORECAST_HISTORY_PATH,
            df[['Date', 'Target', 'Return', 'Close']],
            {
                'date': latest_row['Date'],
                'prediction': int(pred),
                'confidence': confidence,
                'model_name': model_payload['model_name'],
                'previous_direction': int(latest_row['Return'] > 0),
                'predicted_return_pct': predicted_return_pct,
                'reference_close': float(latest_row['Close']),
                'prediction_interval_pct': interval_half_width_pct,
                'model_version': model_payload.get(
                    'trained_at',
                    (
                        f"{model_payload['model_name']}:"
                        f"{model_payload.get('walk_forward', {}).get('test_end_date', 'legacy')}"
                    ),
                ),
            },
        )
        direction = "TĂNG" if pred == 1 else "KHÔNG TĂNG"
        signal_text = (
            f"Tín hiệu mô hình: {direction} · "
            f"xác suất lớp dự báo {proba:.1f}%"
        )
        if pred == 1:
            st.success(signal_text)
        else:
            st.error(signal_text)
        st.metric(
            "Giá đóng cửa ước tính phiên tới",
            f"${expected_close:,.2f}",
            f"{predicted_return_pct:+.3f}%",
        )
        interval_level = model_payload.get(
            'return_metrics',
            {},
        ).get('prediction_interval_level', 0.90)
        st.caption(
            f"Khoảng ước lượng {(interval_level * 100):.0f}%: "
            f"${lower_close:,.2f} – ${upper_close:,.2f}. "
            "Độ rộng lấy từ sai số validation; không phải mức giá đảm bảo."
        )
        st.caption(
            "Xác suất tăng/không tăng là điểm predict_proba chưa hiệu chuẩn; "
            "không nên hiểu là xác suất chính xác ngoài thực tế."
        )
        direction_allowed = (
            telegram_direction_filter == 'Tất cả'
            or (telegram_direction_filter == 'Chỉ tín hiệu tăng' and pred == 1)
            or (telegram_direction_filter == 'Chỉ tín hiệu không tăng' and pred == 0)
        )
        confidence_allowed = proba >= telegram_min_confidence
        if not direction_allowed:
            st.caption("Chưa gửi: hướng tín hiệu không khớp bộ lọc Telegram.")
        elif not confidence_allowed:
            st.caption(
                f"Chưa gửi: độ tin cậy {proba:.1f}% thấp hơn ngưỡng "
                f"{telegram_min_confidence}%."
            )
        if st.button(
            "Gửi tín hiệu này qua Telegram",
            key="send_telegram_signal",
            disabled=not (direction_allowed and confidence_allowed),
        ):
            from src.telegram_bot import send_telegram_message

            telegram_text = (
                "TÍN HIỆU DỰ BÁO GIÁ VÀNG AI\n"
                f"Giá hiện tại: ${latest_row['Close']:,.2f} USD/Oz\n"
                f"Dự báo phiên tới: {direction}\n"
                f"Độ tin cậy của mô hình: {proba:.1f}%\n"
                f"Giá đóng cửa ước tính: ${expected_close:,.2f} USD/Oz\n"
                f"Khoảng ước lượng: ${lower_close:,.2f} - ${upper_close:,.2f} USD/Oz\n"
                "Xác suất phân lớp chưa hiệu chuẩn; khoảng giá không đảm bảo.\n"
                "Chỉ mang tính tham khảo, không phải khuyến nghị đầu tư."
            )
            try:
                send_telegram_message(telegram_text)
            except (RuntimeError, OSError, ValueError) as error:
                st.error(str(error))
            else:
                st.success("Đã gửi tín hiệu Telegram.")
    elif model_payload is not None:
        st.error("Đã tạm ẩn tín hiệu vì dữ liệu có lỗi kiểm tra. Hãy cập nhật dữ liệu rồi thử lại.")
    else:
        st.warning("Chưa có mô hình đã huấn luyện.")

    st.divider()

    if enable_layer1:
        current_vol = latest_row.get('Volatility_20_Ann', 0.0)
        vol_spike = latest_row.get('Volatility_Spike_Flag', 0)
        
        if vol_spike == 1 or current_vol > 20.0:
            st.error("**RỦI RO BIẾN ĐỘNG CAO**\n\n"
                     f"Biến động thường niên: `{current_vol:.1f}%`. "
                     "Cân nhắc giảm quy mô vị thế và đánh giá lại rủi ro trước khi giao dịch.")
        else:
            st.success("**BIẾN ĐỘNG TRONG NGƯỠNG THEO DÕI**\n\n"
                       f"Biến động thường niên hiện tại: `{current_vol:.1f}%`. Đây là chỉ báo tham khảo, không phải khuyến nghị đầu tư.")

    if enable_layer3:
        st.subheader("Tin tức & rủi ro địa chính trị")
        if st.button("Làm mới tin tức", key="refresh_news"):
            st.session_state['news_refresh_key'] = st.session_state.get('news_refresh_key', 0) + 1
        refresh_key = st.session_state.get('news_refresh_key', 0)
        try:
            news_data = load_news(refresh_key)
        except NewsFetchError as error:
            st.error(str(error))
        else:
            n_col1, n_col2 = st.columns([1, 2])
            with n_col1:
                st.metric("Điểm cảm xúc tiêu đề", f"{news_data['avg_sentiment']:.2f}",
                          "Có từ khóa rủi ro" if news_data['crisis_flag'] else "Không phát hiện từ khóa rủi ro")
                if news_data['crisis_flag']:
                    st.warning(f"Có tiêu đề chứa từ khóa rủi ro ({news_data['crisis_count']}).")
                else:
                    st.info("Không phát hiện từ khóa rủi ro trong tiêu đề.")

            with n_col2:
                st.markdown("**Tin thật mới nhất · Google News RSS**")
                translation_available = True
                for item in news_data['news_list']:
                    icon = "🔴" if item['is_crisis'] else "·"
                    article_url = item.get('url', '')
                    original_title = item.get('title', '')
                    translated_title = original_title
                    if translation_available:
                        try:
                            translated_title = translate_news_title(original_title)
                        except NewsTranslationError:
                            translation_available = False
                    if article_url:
                        st.markdown(f"{icon} **{item['source']}** · [{translated_title}]({article_url})")
                    else:
                        st.write(f"{icon} **{item['source']}** · {translated_title}")
                    if translated_title != original_title:
                        st.caption(f"Nguyên văn: {original_title} · {item['published_at']}")
                    else:
                        st.caption(item['published_at'])
                if not translation_available:
                    st.warning("Không thể dịch một số tiêu đề; các tin đó đang hiển thị bằng ngôn ngữ gốc.")
            st.caption("Tin và liên kết bài gốc lấy từ Google News RSS; tiêu đề được dịch sang tiếng Việt, kèm nguyên văn để đối chiếu. Bản dịch máy có thể chưa chính xác tuyệt đối. Bản dịch được lưu tối đa 24 giờ; tin RSS tối đa 15 phút. Điểm cảm xúc chỉ dựa trên từ khóa tiêu đề gốc.")

    st.subheader("Giá vàng và chỉ báo kỹ thuật")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_view['Date'], y=df_view['Close'], mode='lines', name='Giá đóng cửa', line=dict(color='#F2C14E', width=2.4)))
    
    if show_sma:
        for col_name, color in [('SMA_7', '#3B82F6'), ('SMA_14', '#8B5CF6'), ('SMA_50', '#EF4444'), ('SMA_200', '#10B981')]:
            if col_name in df_view.columns:
                fig.add_trace(go.Scatter(x=df_view['Date'], y=df_view[col_name], mode='lines', name=col_name, line=dict(color=color, width=1)))
                
    if show_bb and 'BB_Upper' in df_view.columns and 'BB_Lower' in df_view.columns:
        fig.add_trace(go.Scatter(x=df_view['Date'], y=df_view['BB_Upper'], mode='lines', name='BB Upper', line=dict(color='#9CA3AF', dash='dash')))
        fig.add_trace(go.Scatter(x=df_view['Date'], y=df_view['BB_Lower'], mode='lines', name='BB Lower', line=dict(color='#9CA3AF', dash='dash')))

    fig.update_layout(xaxis_title="Ngày giao dịch", yaxis_title="USD / oz", hovermode="x unified", height=440)
    style_chart(fig)
    st.plotly_chart(fig, width="stretch")

    if show_daily_vol and 'Daily_Return_Pct' in df_view.columns:
        st.subheader("Lợi suất theo phiên")
        colors = ['#54C99A' if val >= 0 else '#F07178' for val in df_view['Daily_Return_Pct']]
        fig_d = go.Figure()
        fig_d.add_trace(go.Bar(x=df_view['Date'], y=df_view['Daily_Return_Pct'], marker_color=colors, name="Lợi suất (%)"))
        fig_d.update_layout(xaxis_title="Ngày", yaxis_title="Lợi suất (%)", height=300, hovermode="x unified")
        style_chart(fig_d)
        st.plotly_chart(fig_d, width="stretch")
    st.download_button(
        "Tải CSV dữ liệu đang xem",
        data=df_view.to_csv(index=False).encode('utf-8-sig'),
        file_name="gold_market_data.csv",
        mime="text/csv",
    )

# ================= TAB 4: VIETNAM SJC QUOTES =================
with tab4:
    st.subheader("Giá vàng miếng SJC Việt Nam")
    st.caption(
        "Giá SJC được nhập từ tệp CSV do bạn cung cấp. Ứng dụng không tự giả lập "
        "giá SJC và chưa kết nối nguồn báo giá trực tiếp. Hãy xác minh nguồn và thời điểm cập nhật."
    )
    st.markdown(
        "Tệp CSV cần các cột `Date`, `Buy_VND_per_luong`, `Sell_VND_per_luong`, `Source`. "
        "Ngày dùng định dạng YYYY-MM-DD; giá tính bằng VND/lượng."
    )
    template_csv = "Date,Buy_VND_per_luong,Sell_VND_per_luong,Source\n"
    st.download_button(
        "Tải mẫu CSV giá SJC", data=template_csv.encode("utf-8-sig"),
        file_name="sjc_prices_template.csv", mime="text/csv", key="download_sjc_template",
    )
    sjc_upload = st.file_uploader("Chọn CSV báo giá SJC", type=["csv"], key="sjc_csv_upload")
    sjc_frame = None
    if sjc_upload is not None:
        try:
            sjc_frame = validate_sjc_quotes(pd.read_csv(sjc_upload))
        except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as error:
            st.error(f"Không thể đọc báo giá SJC: {error}")
    elif os.path.exists(SJC_DATA_PATH):
        try:
            sjc_frame = validate_sjc_quotes(pd.read_csv(SJC_DATA_PATH))
        except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as error:
            st.error(f"Tệp SJC đã lưu không hợp lệ: {error}")

    if sjc_frame is not None and not sjc_frame.empty:
        if sjc_upload is not None and st.button("Lưu dữ liệu SJC vào ứng dụng", key="save_sjc_quotes"):
            try:
                with atomic_output_path(SJC_DATA_PATH) as temporary_path:
                    sjc_frame.to_csv(temporary_path, index=False, date_format="%Y-%m-%d")
            except OSError as error:
                st.error(f"Không thể lưu dữ liệu SJC: {error}")
            else:
                st.success("Đã lưu báo giá SJC đã kiểm tra vào data/sjc_prices.csv.")
        latest_sjc = sjc_frame.iloc[-1]
        quote_date = pd.Timestamp(latest_sjc["Date"]).date()
        age_days = max((pd.Timestamp.now().date() - quote_date).days, 0)
        if age_days > 3:
            st.warning(f"Báo giá mới nhất đã {age_days} ngày tuổi; hãy kiểm tra lại nguồn trước khi sử dụng.")
        else:
            st.info(f"Ngày báo giá mới nhất: {quote_date:%d/%m/%Y} · Nguồn: {latest_sjc['Source']}")

        sjc_buy, sjc_sell, sjc_spread = st.columns(3)
        sjc_buy.metric("SJC mua vào", f"{latest_sjc['Buy_VND_per_luong']:,.0f} ₫/lượng")
        sjc_sell.metric("SJC bán ra", f"{latest_sjc['Sell_VND_per_luong']:,.0f} ₫/lượng")
        sjc_spread.metric(
            "Chênh lệch mua–bán",
            f"{latest_sjc['Sell_VND_per_luong'] - latest_sjc['Buy_VND_per_luong']:,.0f} ₫",
        )
        sjc_chart = go.Figure()
        sjc_chart.add_trace(go.Scatter(
            x=sjc_frame["Date"], y=sjc_frame["Buy_VND_per_luong"],
            mode="lines+markers", name="Mua vào",
        ))
        sjc_chart.add_trace(go.Scatter(
            x=sjc_frame["Date"], y=sjc_frame["Sell_VND_per_luong"],
            mode="lines+markers", name="Bán ra",
        ))
        sjc_chart.update_layout(title="Lịch sử báo giá SJC", xaxis_title="Ngày", yaxis_title="VND/lượng", height=390)
        style_chart(sjc_chart)
        st.plotly_chart(sjc_chart, width="stretch", key="sjc_history_chart")

        st.markdown("**So sánh tham khảo với vàng thế giới**")
        fx = st.number_input(
            "Tỷ giá USD/VND do bạn nhập", min_value=0.0, value=0.0, step=100.0,
            help="Nhập tỷ giá tham chiếu từ nguồn bạn tin cậy. Không tự động cập nhật và không phải báo giá ngân hàng trực tiếp.",
            key="sjc_usd_vnd_rate",
        )
        if fx > 0 and float(latest_row["Close"]) > 0:
            try:
                comparison = quote_comparison(sjc_frame, float(latest_row["Close"]), fx)
            except ValueError as error:
                st.error(str(error))
            else:
                eq_col, premium_col, pct_col = st.columns(3)
                eq_col.metric("Giá quốc tế quy đổi*", f"{comparison['international_equivalent']:,.0f} ₫/lượng")
                premium_col.metric("Chênh lệch so với giá SJC giữa mua–bán", f"{comparison['midpoint_premium_vnd']:+,.0f} ₫")
                pct_col.metric("Chênh lệch tương đối", f"{comparison['midpoint_premium_pct']:+.2f}%")
                st.caption(
                    "*Quy đổi lý thuyết từ GC=F (USD/ounce troy) và tỷ giá bạn nhập, "
                    "chưa gồm thuế, phí, chi phí nhập khẩu, khác biệt sản phẩm hay điều kiện thị trường. "
                    "Không dùng làm giá giao dịch hoặc khuyến nghị đầu tư."
                )
        else:
            st.info("Nhập tỷ giá USD/VND để xem phép quy đổi tham khảo; ứng dụng không tự đoán tỷ giá hiện hành.")
        st.dataframe(sjc_frame.sort_values("Date", ascending=False), hide_index=True, width="stretch")
        st.download_button(
            "Tải dữ liệu SJC đã kiểm tra", data=sjc_frame.to_csv(index=False).encode("utf-8-sig"),
            file_name="sjc_prices_validated.csv", mime="text/csv", key="download_sjc_validated",
        )
    else:
        st.info(
            "Chưa có dữ liệu SJC. Tải mẫu CSV, điền báo giá có nguồn rõ ràng rồi tải lên. "
            "Các giá trị mẫu không được tự tạo để tránh hiển thị giá sai."
        )

# ================= TAB 2: BACKTEST QUANT =================
with tab2:
    st.subheader("Hiệu suất mô phỏng")
    st.caption("Backtest dùng duy nhất dự báo ngoài mẫu theo walk-forward; kết quả lịch sử không đảm bảo hiệu suất tương lai.")

    if model_payload is not None and model_payload.get('oos_predictions'):
        try:
            test_data, test_signals = prepare_walk_forward_backtest_data(
                df,
                model_payload['oos_predictions'],
            )
        except (KeyError, TypeError, ValueError) as error:
            st.warning(f"Không thể dùng kết quả kiểm định hiện tại: {error}")
        else:
            col_set1, col_set2, col_set3, col_set4, col_set5 = st.columns(5)
            initial_capital = col_set1.number_input("Số vốn ban đầu (USD):", value=10000, step=1000)
            stop_loss_pct = col_set2.slider("Cắt lỗ tự động (Stop-Loss %):", min_value=0.5, max_value=5.0, value=1.5, step=0.1) / 100
            take_profit_pct = col_set3.slider("Chốt lời tự động (Take-Profit %):", min_value=1.0, max_value=10.0, value=3.0, step=0.5) / 100
            transaction_cost_bps = col_set4.number_input("Phí mỗi chiều (bps):", min_value=0.0, max_value=100.0, value=0.0, step=0.5)
            slippage_bps = col_set5.number_input("Trượt giá mỗi chiều (bps):", min_value=0.0, max_value=100.0, value=0.0, step=0.5)
            st.caption(
                "Phí và trượt giá được nhập theo basis point (1 bps = 0,01%), "
                "áp dụng cho cả chiều vào và chiều thoát lệnh. Tín hiệu được giữ tối đa một phiên."
            )

            test_df = run_backtest(
                test_data=test_data,
                signals=test_signals,
                initial_capital=initial_capital,
                stop_loss_pct=stop_loss_pct,
                take_profit_pct=take_profit_pct,
                transaction_cost_bps=transaction_cost_bps,
                slippage_bps=slippage_bps,
                volatility_position_sizing=enable_layer2,
            )
            trade_returns = test_df.loc[test_df['Signal'] == 1, 'Net_Return']
            win_trades = int((trade_returns > 0).sum())

            ai_final = test_df['AI_Portfolio'].iloc[-1]
            ai_return_pct = ((ai_final - initial_capital) / initial_capital) * 100

            portfolio_series = pd.Series(test_df['AI_Portfolio'])
            rolling_max = portfolio_series.cummax()
            drawdowns = (portfolio_series - rolling_max) / rolling_max
            max_drawdown = drawdowns.min() * 100

            daily_returns = portfolio_series.pct_change().dropna()
            sharpe_ratio = (
                (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
                if len(daily_returns) > 0 and daily_returns.std() != 0 else 0.0
            )

            total_trades = int((test_df['Signal'].iloc[:-1] == 1).sum())
            win_rate = (win_trades / total_trades * 100) if total_trades > 0 else 0.0

            buy_hold_ret = (test_df['Close'] - test_df['Close'].iloc[0]) / test_df['Close'].iloc[0]
            test_df['Buy_Hold_Portfolio'] = initial_capital * (1 + buy_hold_ret)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Tài Sản Cuối Kỳ (AI)", f"${ai_final:,.2f}", f"{ai_return_pct:+.2f}%")
            c2.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}", "Hiệu quả lợi nhuận/rủi ro")
            c3.metric("Max Drawdown (MDD)", f"{max_drawdown:.2f}%", "Mức sụt giảm tối đa")
            c4.metric("Win Rate (%)", f"{win_rate:.1f}%", f"{win_trades}/{total_trades} Lệnh thắng")

            fig_bt = go.Figure()
            fig_bt.add_trace(go.Scatter(x=test_df['Date'], y=test_df['AI_Portfolio'], name='Chiến Lược Quant AI (Bảo vệ 3 Tầng)', line=dict(color='#10B981', width=2.5)))
            fig_bt.add_trace(go.Scatter(x=test_df['Date'], y=test_df['Buy_Hold_Portfolio'], name='Chiến Lược Buy & Hold', line=dict(color='#6B7280', dash='dash')))
            fig_bt.update_layout(xaxis_title="Ngày", yaxis_title="Vốn mô phỏng (USD)", height=400)
            style_chart(fig_bt)
            st.plotly_chart(fig_bt, width="stretch")
            st.download_button(
                "Tải CSV kết quả backtest",
                data=test_df.to_csv(index=False).encode('utf-8-sig'),
                file_name="gold_backtest_results.csv",
                mime="text/csv",
            )
    elif model_payload is not None:
        st.info("Mô hình hiện tại chưa có kết quả walk-forward. Chạy lại `python src/train.py` để tạo đánh giá ngoài mẫu.")
    else:
        st.info("Huấn luyện mô hình để mở phần mô phỏng hiệu suất.")

# ================= TAB 3: MODEL DIAGNOSTICS =================
with tab3:
    st.subheader("Chất lượng mô hình")
    st.caption("Các chỉ số chính được tính trên dự báo ngoài mẫu theo walk-forward; các mô hình chỉ được chọn bằng validation.")
    if model_payload is not None:
        metrics = model_payload['metrics']
        st.markdown(f"**Mô hình đang sử dụng** · `{model_payload['model_name']}`")
        walk_forward = model_payload.get('walk_forward', {})
        if walk_forward:
            st.info(
                f"Chia theo thời gian: train {walk_forward['train_size']} phiên đến "
                f"{walk_forward['train_end_date']}; validation {walk_forward['validation_size']} phiên đến "
                f"{walk_forward['validation_end_date']}; walk-forward test {walk_forward['test_size']} phiên "
                f"từ {walk_forward['test_start_date']} đến {walk_forward['test_end_date']} "
                f"({walk_forward['fold_count']} đợt, mỗi đợt tối đa {walk_forward['step_size']} phiên)."
            )

        return_metrics = model_payload.get('return_metrics')
        if return_metrics:
            st.markdown("**Đánh giá dự báo mức biến động phiên kế tiếp**")
            return_columns = st.columns(4)
            return_columns[0].metric(
                "MAE walk-forward",
                f"{return_metrics['mae_pct']:.3f}%",
            )
            return_columns[1].metric(
                "RMSE walk-forward",
                f"{return_metrics['rmse_pct']:.3f}%",
            )
            return_columns[2].metric(
                "Đúng hướng dự báo lợi suất",
                f"{return_metrics['directional_accuracy'] * 100:.2f}%",
            )
            return_columns[3].metric(
                "Độ phủ khoảng ước lượng",
                f"{return_metrics['prediction_interval_coverage'] * 100:.1f}%",
                f"mục tiêu {return_metrics['prediction_interval_level'] * 100:.0f}%",
            )
            st.caption(
                f"Mô hình lợi suất được chọn trên validation: "
                f"{model_payload['return_model_name']}. Khoảng ước lượng được xác định "
                "từ phân vị 90% sai số tuyệt đối validation; độ phủ test được báo cáo "
                "riêng, không dùng test để điều chỉnh khoảng."
            )
            return_baseline_metrics = model_payload.get('return_baseline_metrics')
            if return_baseline_metrics:
                return_comparison = pd.DataFrame([
                    {
                        'Đánh giá trên test walk-forward': model_payload['return_model_name'],
                        'MAE': return_metrics['mae_pct'],
                        'RMSE': return_metrics['rmse_pct'],
                        'Đúng hướng': return_metrics['directional_accuracy'],
                    },
                    {
                        'Đánh giá trên test walk-forward': 'Baseline dự báo lợi suất 0%',
                        'MAE': return_baseline_metrics['mae_pct'],
                        'RMSE': return_baseline_metrics['rmse_pct'],
                        'Đúng hướng': return_baseline_metrics['directional_accuracy'],
                    },
                ])
                return_comparison['MAE'] = return_comparison['MAE'].map(
                    lambda value: f"{value:.4f}%"
                )
                return_comparison['RMSE'] = return_comparison['RMSE'].map(
                    lambda value: f"{value:.4f}%"
                )
                return_comparison['Đúng hướng'] = return_comparison['Đúng hướng'].map(
                    lambda value: f"{value * 100:.2f}%"
                )
                st.dataframe(
                    return_comparison,
                    hide_index=True,
                    width="stretch",
                    alt="So sánh dự báo lợi suất với baseline không đổi trên test walk-forward",
                )
            validation_return_metrics = model_payload.get(
                'return_validation_metrics',
                {},
            )
            if validation_return_metrics:
                validation_return_table = pd.DataFrame.from_dict(
                    validation_return_metrics,
                    orient='index',
                ).rename_axis('Ứng viên dự báo lợi suất').reset_index()
                validation_return_table['mae_pct'] = validation_return_table[
                    'mae_pct'
                ].map(lambda value: f"{value:.4f}%")
                validation_return_table['rmse_pct'] = validation_return_table[
                    'rmse_pct'
                ].map(lambda value: f"{value:.4f}%")
                validation_return_table['directional_accuracy'] = validation_return_table[
                    'directional_accuracy'
                ].map(lambda value: f"{value * 100:.2f}%")
                st.dataframe(
                    validation_return_table.rename(columns={
                        'mae_pct': 'MAE',
                        'rmse_pct': 'RMSE',
                        'directional_accuracy': 'Đúng hướng',
                    }),
                    hide_index=True,
                    width="stretch",
                    alt="So sánh mô hình dự báo lợi suất trên tập validation",
                )

        st.subheader("Theo dõi dự báo thực tế")
        st.caption(
            "Dự báo được lưu cục bộ theo ngày và phiên bản mô hình; kết quả chỉ được chấm "
            "khi dữ liệu giá của phiên kế tiếp đã có. Xác suất phân lớp chưa hiệu chuẩn."
        )
        tracking = summarize_forecast_history(forecast_history)
        tracking_window = st.selectbox(
            "Khoảng thời gian theo dõi",
            ['30 ngày gần nhất', '90 ngày gần nhất', 'Toàn bộ lịch sử'],
            index=2,
            key='forecast_tracking_window',
        )
        if tracking_window != 'Toàn bộ lịch sử' and not forecast_history.empty:
            window_days = 30 if tracking_window.startswith('30') else 90
            history_dates = pd.to_datetime(forecast_history['forecast_date'])
            latest_forecast_date = history_dates.max()
            forecast_history = forecast_history.loc[
                history_dates >= latest_forecast_date - pd.Timedelta(days=window_days - 1)
            ].copy()
            tracking = summarize_forecast_history(forecast_history)
        if tracking['resolved_count']:
            tracking_columns = st.columns(4)
            tracking_columns[0].metric("Dự báo đã chấm", tracking['resolved_count'])
            tracking_columns[1].metric(
                "Độ chính xác thực tế",
                f"{tracking['accuracy'] * 100:.2f}%",
            )
            tracking_columns[2].metric(
                "F1 thực tế",
                f"{tracking['f1_score'] * 100:.2f}%",
            )
            tracking_columns[3].metric(
                "Điểm xác suất trung bình",
                f"{tracking['average_confidence'] * 100:.1f}%",
            )
            tracked_baselines = tracking['baseline_metrics']
            if tracked_baselines:
                st.markdown(
                    "**Đối chiếu F1 trên cùng "
                    f"{tracked_baselines['sample_count']} phiên đã chấm**"
                )
                baseline_columns = st.columns(3)
                baseline_columns[0].metric(
                    "Mô hình · F1",
                    f"{tracked_baselines['model']['f1_score'] * 100:.2f}%",
                )
                baseline_columns[1].metric(
                    "Luôn dự báo tăng · F1",
                    f"{tracked_baselines['always_up']['f1_score'] * 100:.2f}%",
                )
                baseline_columns[2].metric(
                    "Theo hướng phiên trước · F1",
                    f"{tracked_baselines['previous_day_direction']['f1_score'] * 100:.2f}%",
                )
                if (
                    tracked_baselines['model']['f1_score']
                    < tracked_baselines['always_up']['f1_score']
                ):
                    st.warning(
                        "Trên các dự báo đã chấm, F1 của mô hình thấp hơn baseline luôn dự báo tăng."
                    )
            tracked_returns = tracking['return_metrics']
            if tracked_returns:
                st.markdown(
                    f"**Sai số giá/lợi suất dự báo thực tế · "
                    f"{tracked_returns['sample_count']} phiên**"
                )
                return_tracking_columns = st.columns(4)
                return_tracking_columns[0].metric(
                    "MAE lợi suất",
                    f"{tracked_returns['mae_pct']:.3f}%",
                )
                return_tracking_columns[1].metric(
                    "MAE baseline 0%",
                    f"{tracked_returns['zero_baseline_mae_pct']:.3f}%",
                )
                return_tracking_columns[2].metric(
                    "RMSE lợi suất",
                    f"{tracked_returns['rmse_pct']:.3f}%",
                )
                return_tracking_columns[3].metric(
                    "Đúng hướng",
                    f"{tracked_returns['directional_accuracy'] * 100:.1f}%",
                )
                if tracked_returns['mae_pct'] >= tracked_returns['zero_baseline_mae_pct']:
                    st.warning("Lợi suất dự báo chưa tốt hơn baseline 0% trên giai đoạn đã chọn.")
                tracked_returns_history = tracking['resolved_history'].dropna(
                    subset=['predicted_return_pct', 'actual_return_pct']
                )
                if not tracked_returns_history.empty:
                    return_chart_data = tracked_returns_history.melt(
                        id_vars=['forecast_date'],
                        value_vars=['predicted_return_pct', 'actual_return_pct'],
                        var_name='Loại',
                        value_name='Lợi suất (%)',
                    )
                    return_chart_data['Loại'] = return_chart_data['Loại'].map({
                        'predicted_return_pct': 'Dự báo',
                        'actual_return_pct': 'Thực tế',
                    })
                    return_chart = px.line(
                        return_chart_data,
                        x='forecast_date',
                        y='Lợi suất (%)',
                        color='Loại',
                        title='Lợi suất dự báo và thực tế theo phiên',
                    )
                    style_chart(return_chart)
                    st.plotly_chart(
                        return_chart,
                        width='stretch',
                        key='forecast_return_tracking',
                    )
            else:
                st.info(
                    "Chưa có đủ phiên để chấm lợi suất/giá dự báo trong khoảng đã chọn. "
                    "Kết quả sẽ xuất hiện sau khi có giá đóng cửa phiên kế tiếp."
                )
            resolved_history = tracking['resolved_history']
            resolved_history['Độ chính xác lũy kế (%)'] = (
                resolved_history['predicted_direction']
                .eq(resolved_history['actual_direction'])
                .expanding()
                .mean()
                .mul(100)
            )
            accuracy_chart = px.line(
                resolved_history,
                x='forecast_date',
                y='Độ chính xác lũy kế (%)',
                title='Độ chính xác lũy kế theo ngày dự báo',
            )
            accuracy_chart.update_layout(height=320, yaxis_range=[0, 100])
            style_chart(accuracy_chart)
            st.plotly_chart(
                accuracy_chart,
                width="stretch",
                key="forecast_accuracy_history",
            )
        else:
            st.info("Chưa có dự báo nào được đối chiếu với giá thực tế.")
        if tracking['pending_count']:
            st.caption(
                f"{tracking['pending_count']} dự báo đang chờ dữ liệu giá của phiên kế tiếp."
            )
        if tracking['forecast_count']:
            direction_names = {0: 'KHÔNG TĂNG', 1: 'TĂNG'}
            display_history = tracking['latest_history'].sort_values(
                'forecast_date',
                ascending=False,
            ).head(20).copy()
            display_history['Dự báo'] = display_history['predicted_direction'].map(direction_names)
            display_history['Thực tế'] = display_history['actual_direction'].map(direction_names)
            display_history['Điểm xác suất'] = display_history['confidence'].map(
                lambda value: f"{value * 100:.1f}%"
            )
            for column, label in (
                ('predicted_return_pct', 'Lợi suất dự báo'),
                ('actual_return_pct', 'Lợi suất thực tế'),
            ):
                missing_label = (
                    'Không lưu ở phiên bản cũ'
                    if column.startswith('predicted') else 'Chờ phiên kế'
                )
                if column not in display_history:
                    display_history[label] = missing_label
                else:
                    display_history[label] = display_history[column].map(
                        lambda value: (
                            f"{value:+.3f}%" if pd.notna(value) else missing_label
                        )
                    )
            for column, label in (
                ('predicted_close', 'Giá ước tính'),
                ('actual_close', 'Giá thực tế'),
            ):
                missing_label = (
                    'Không lưu ở phiên bản cũ'
                    if column.startswith('predicted') else 'Chờ phiên kế'
                )
                if column not in display_history:
                    display_history[label] = missing_label
                else:
                    display_history[label] = display_history[column].map(
                        lambda value: (
                            f"${value:,.2f}" if pd.notna(value) else missing_label
                        )
                    )
            display_history['Ngày dự báo'] = pd.to_datetime(
                display_history['forecast_date']
            ).dt.strftime('%d/%m/%Y')
            history_columns = [
                'Ngày dự báo',
                'Dự báo',
                'Thực tế',
                'Điểm xác suất',
                'Lợi suất dự báo',
                'Lợi suất thực tế',
                'Giá ước tính',
                'Giá thực tế',
                'model_name',
            ]
            visible_history = display_history[history_columns].rename(
                columns={'model_name': 'Mô hình'}
            )
            st.dataframe(
                visible_history,
                hide_index=True,
                width="stretch",
                alt="Lịch sử dự báo hướng giá vàng và kết quả thực tế",
            )
            st.download_button(
                "Tải lịch sử dự báo dạng CSV",
                data=visible_history.to_csv(index=False).encode('utf-8-sig'),
                file_name='lich_su_du_bao_vang.csv',
                mime='text/csv',
                width='stretch',
            )
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
        m2.metric("Precision", f"{metrics['precision']*100:.2f}%")
        m3.metric("Recall", f"{metrics['recall']*100:.2f}%")
        m4.metric("F1-Score", f"{metrics['f1_score']*100:.2f}%")

        baseline_metrics = model_payload.get('baseline_metrics', {})
        if baseline_metrics:
            st.markdown("**Đối chiếu trên cùng giai đoạn test**")
            baseline_columns = st.columns(3)
            baseline_columns[0].metric("Mô hình · F1", f"{metrics['f1_score'] * 100:.2f}%")
            baseline_columns[1].metric(
                "Luôn dự báo tăng · F1",
                f"{baseline_metrics['always_up']['f1_score'] * 100:.2f}%",
            )
            baseline_columns[2].metric(
                "Theo hướng phiên trước · F1",
                f"{baseline_metrics['previous_day_direction']['f1_score'] * 100:.2f}%",
            )
            if metrics['f1_score'] < baseline_metrics['always_up']['f1_score']:
                st.warning(
                    "F1 của mô hình hiện thấp hơn baseline luôn dự báo tăng. "
                    "Cần cải thiện và kiểm định thêm trước khi xem tín hiệu là có lợi thế."
                )
        
        st.divider()
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("**10 đặc trưng quan trọng nhất**")
            st.caption(
                "Đây là tầm quan trọng tổng thể trong mô hình, không chứng minh nguyên nhân "
                "hay giải thích riêng cho tín hiệu hôm nay."
            )
            importances = model_payload.get('feature_importances', {})
            if importances:
                top_imp = dict(list(importances.items())[:10])
                fig_imp = px.bar(x=list(top_imp.values()), y=list(top_imp.keys()), orientation='h', color=list(top_imp.values()), color_continuous_scale='Viridis')
                fig_imp.update_layout(yaxis=dict(autorange="reversed"), height=380, showlegend=False)
                style_chart(fig_imp)
                st.plotly_chart(fig_imp, width="stretch")
        with col_m2:
            st.markdown("**Ma trận nhầm lẫn**")
            cm = np.array(metrics['confusion_matrix'])
            fig_cm = px.imshow(cm, text_auto=True, x=['Dự báo KHÔNG TĂNG', 'Dự báo TĂNG'], y=['Thực tế KHÔNG TĂNG', 'Thực tế TĂNG'], color_continuous_scale='Blues')
            fig_cm.update_layout(height=380)
            style_chart(fig_cm)
            st.plotly_chart(fig_cm, width="stretch")
   
    else:
        st.info(
            "Chưa có mô hình. Chạy `python src/train.py` "
            "để huấn luyện và lưu mô hình."
        )

# ================= TAB 5: PROJECT SUMMARY =================
with tab5:
    st.subheader("Tổng quan và phương pháp thực hiện")
    st.caption(
        "Trang tóm tắt được tạo từ cấu hình và kết quả mô hình hiện có, "
        "có thể dùng làm nội dung tham khảo khi thuyết trình."
    )
    project_summary = build_project_summary(
        latest_row["Date"].strftime("%d/%m/%Y"),
        len(df),
        model_payload,
    )
    summary_columns = st.columns(3)
    summary_columns[0].metric(
        "Phiên dữ liệu mới nhất",
        latest_row["Date"].strftime("%d/%m/%Y"),
    )
    summary_columns[1].metric(
        "Số phiên dữ liệu",
        f"{len(df):,}",
    )
    summary_columns[2].metric(
        "Mô hình phân loại",
        model_payload.get("model_name", "Chưa huấn luyện")
        if model_payload
        else "Chưa huấn luyện",
    )
    st.markdown(project_summary)
    st.download_button(
        "Tải tóm tắt kỹ thuật để báo cáo",
        data=project_summary.encode("utf-8-sig"),
        file_name="tom_tat_do_an_du_bao_gia_vang.md",
        mime="text/markdown",
        key="download_project_summary",
    )
