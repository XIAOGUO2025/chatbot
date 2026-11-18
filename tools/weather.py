# tools/weather.py
import requests
from config import WEATHER_KEYS, EXCLUDE_WEATHER_FOLLOWUP, DEFAULT_CITY

WMO_ZH = {0: "晴", 1: "基本晴", 2: "多云", 3: "阴", 45: "雾", 48: "雾凇", 51: "小毛毛雨", 53: "毛毛雨", 55: "大毛毛雨", 61: "小雨", 63: "中雨", 65: "大雨", 66: "冻雨", 67: "强冻雨", 71: "小雪", 73: "中雪", 75: "大雪", 77: "雪粒", 80: "小阵雨", 81: "阵雨", 82: "强阵雨", 85: "小阵雪", 86: "大阵雪", 95: "雷阵雨", 96: "雷阵雨伴小冰雹", 99: "雷阵雨伴大冰雹"}

def geocode_city(city):
    try:
        r = requests.get("https://geocoding-api.open-meteo.com/v1/search", params={"name": city, "count": 1, "language": "zh"}, timeout=6).json()
        res = r.get("results")
        return (res[0]["name"], res[0]["latitude"], res[0]["longitude"]) if res else None
    except Exception:
        return None

def get_weather(city):
    geo = geocode_city(city)
    if not geo: return None
    name, lat, lon = geo
    try:
        r = requests.get("https://api.open-meteo.com/v1/forecast", params={"latitude": lat, "longitude": lon, "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code", "timezone": "auto"}, timeout=6).json()
        cur = r["current"]
        return {"city": name, "temp": cur["temperature_2m"], "humidity": cur["relative_humidity_2m"], "wind": cur["wind_speed_10m"], "code": cur["weather_code"]}
    except Exception:
        return None

def is_weather_intent(text):
    return any(k in text for k in WEATHER_KEYS)

def extract_city(text):
    known = ["北京", "上海", "广州", "深圳", "杭州", "成都", "武汉", "西安", "南京", "重庆", "天津", "苏州", "长沙", "郑州", "沈阳", "青岛", "大连", "厦门", "昆明", "合肥"]
    for c in known:
        if c in text: return c
    for key in ["天气", "气温", "温度"]:
        idx = text.find(key)
        if idx > 0:
            prefix = text[:idx].strip("，,。.！!？? ")
            if 1 <= len(prefix) <= 6: return prefix
    return None

def extract_city_with_ctx(text, session=None):
    city = extract_city(text)
    if city: return city
    if session:
        for prev in reversed(session.recent_user_texts(3)):
            c = extract_city(prev)
            if c: return c
    return None

def is_weather_followup(text, session=None):
    if session is None or not session.history: return False
    prev_users = session.recent_user_texts(2)
    if not prev_users or not any(is_weather_intent(p) for p in prev_users): return False
    if any(w in text for w in EXCLUDE_WEATHER_FOLLOWUP): return False
    if extract_city(text): return len(text) <= 12
    if len(text) <= 8 and any(p in text for p in ["那", "呢", "怎么样", "如何", "多少"]): return True
    return False

def answer_weather(text, session=None, default_city=DEFAULT_CITY):
    city = extract_city_with_ctx(text, session) or default_city
    info = get_weather(city)
    if not info: return f"抱歉，我暂时查不到“{city}”的天气，换个城市名试试？"
    desc = WMO_ZH.get(info["code"], "未知天气")
    return f"{info['city']}现在{desc}，气温 {info['temp']}℃，湿度 {info['humidity']}%，风速 {info['wind']} km/h。"