
import json
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def call_api(url):
    """Gọi API và chuyển kết quả JSON thành dictionary."""
    request = Request(
        url,
        headers={"User-Agent": "RiceLeafAI/1.0"}
    )

    with urlopen(request, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def search_location(location_name):
    """Tìm địa điểm và tọa độ ruộng lúa."""

    params = {
        "name": location_name,
        "count": 10,
        "language": "vi",
        "format": "json",
        "countryCode": "VN"
    }

    url = (
        "https://geocoding-api.open-meteo.com/v1/search?"
        + urlencode(params)
    )

    data = call_api(url)
    return data.get("results", [])


def get_weather(latitude, longitude):
    """Lấy dự báo thời tiết 5 ngày tại vị trí ruộng."""

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "precipitation"
        ),
        "daily": (
            "temperature_2m_max,"
            "temperature_2m_min,"
            "precipitation_sum,"
            "precipitation_probability_max"
        ),
        "forecast_days": 5,
        "timezone": "auto"
    }

    url = (
        "https://api.open-meteo.com/v1/forecast?"
        + urlencode(params)
    )

    return call_api(url)
