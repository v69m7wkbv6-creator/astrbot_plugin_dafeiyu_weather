"""大肥鱼天气 · 核心逻辑。

这一层不依赖 AstrBot，方便单独测试：
- 免费天气源适配（中国天气网 / Open-Meteo / wttr.in）
- 多源自动故障切换、冷却、缓存
- 中文地名解析（"临沂" -> 城市代码 + 经纬度）
- 天气意图识别、时间范围/关注点识别
- 用户地区绑定（JSON 持久化）
- 无人格依赖的兜底模板

只使用 Python 标准库。
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import gzip
import json
import os
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable

__all__ = [
    "Location",
    "DailyForecast",
    "WeatherData",
    "WeatherError",
    "ProviderError",
    "LocationError",
    "BindingStore",
    "WeatherService",
    "detect_weather_intent",
    "detect_time_scope",
    "detect_aspects",
    "extract_city_candidates",
    "clean_location_answer",
    "is_cancel",
    "looks_like_location",
    "render_template",
    "build_ask_prompt",
    "build_reply_prompt",
    "clean_llm_reply",
    "ask_location_template",
    "advice_lines",
    "format_temp",
]

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


# ---------------------------------------------------------------------------
# 异常
# ---------------------------------------------------------------------------


class WeatherError(Exception):
    """天气相关的可预期错误。"""


class ProviderError(WeatherError):
    """单个数据源失败。"""


class LocationError(WeatherError):
    """地点无法解析。"""


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


def _http_get(
    url: str,
    *,
    timeout: float,
    retries: int = 1,
    headers: dict[str, str] | None = None,
) -> tuple[bytes, str]:
    """同步 GET，返回 (bytes, charset)。会处理 gzip 魔数。"""
    last: Exception | None = None
    merged = {
        "User-Agent": USER_AGENT,
        "Accept": "*/*",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.6",
        "Connection": "close",
    }
    if headers:
        merged.update(headers)

    attempts = max(0, int(retries)) + 1
    for attempt in range(attempts):
        try:
            req = urllib.request.Request(url, headers=merged)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
                charset = ""
                try:
                    charset = resp.headers.get_content_charset() or ""
                except Exception:
                    charset = ""
                status = getattr(resp, "status", 200)
            if status and status >= 400:
                raise ProviderError(f"HTTP {status}")
            if raw[:2] == b"\x1f\x8b":
                raw = gzip.decompress(raw)
            return raw, charset
        except Exception as exc:  # noqa: BLE001 - 统一转成 ProviderError
            last = exc
            if attempt + 1 < attempts:
                time.sleep(0.35)
    raise ProviderError(f"请求失败：{type(last).__name__}: {last}")


def _decode(raw: bytes, charset: str = "") -> str:
    candidates = [charset, "utf-8", "gb18030", "gbk"]
    for enc in candidates:
        if not enc:
            continue
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("utf-8", errors="replace")


def _get_json(
    url: str,
    *,
    timeout: float,
    retries: int = 1,
    headers: dict[str, str] | None = None,
) -> Any:
    raw, charset = _http_get(url, timeout=timeout, retries=retries, headers=headers)
    text = _decode(raw, charset)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProviderError(f"返回的不是 JSON：{text[:80]!r}") from exc


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Location:
    name: str
    province: str = ""
    lat: float | None = None
    lon: float | None = None
    cn_code: str = ""

    @property
    def key(self) -> str:
        if self.lat is not None and self.lon is not None:
            return f"{self.lat:.3f},{self.lon:.3f}"
        if self.cn_code:
            return f"cn:{self.cn_code}"
        return f"name:{self.name}"

    @property
    def display(self) -> str:
        if self.province and not self.name.startswith(self.province):
            return f"{self.province}{self.name}"
        return self.name

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "province": self.province,
            "lat": self.lat,
            "lon": self.lon,
            "cn_code": self.cn_code,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Location":
        return cls(
            name=str(data.get("name") or "").strip(),
            province=str(data.get("province") or "").strip(),
            lat=_to_float(data.get("lat")),
            lon=_to_float(data.get("lon")),
            cn_code=str(data.get("cn_code") or "").strip(),
        )


@dataclass
class DailyForecast:
    date: str = ""
    label: str = ""
    text: str = ""
    text_night: str = ""
    temp_max: float | None = None
    temp_min: float | None = None
    precip_prob: int | None = None
    wind_dir: str = ""
    wind_scale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date,
            "label": self.label,
            "text": self.text,
            "text_night": self.text_night,
            "temp_max": self.temp_max,
            "temp_min": self.temp_min,
            "precip_prob": self.precip_prob,
            "wind_dir": self.wind_dir,
            "wind_scale": self.wind_scale,
        }


@dataclass
class HourlyPoint:
    time: str = ""
    temp: float | None = None
    text: str = ""
    precip_prob: int | None = None
    precip_mm: float | None = None

    @property
    def hour(self) -> int:
        match = re.search(r"(\d{1,2})", self.time or "")
        return int(match.group(1)) if match else -1

    def to_dict(self) -> dict[str, Any]:
        return {
            "time": self.time,
            "temp": self.temp,
            "text": self.text,
            "precip_prob": self.precip_prob,
            "precip_mm": self.precip_mm,
        }


@dataclass
class WeatherData:
    location: Location
    source: str = ""
    source_label: str = ""
    current_text: str = ""
    current_temp: float | None = None
    feels_like: float | None = None
    humidity: int | None = None
    wind_dir: str = ""
    wind_scale: str = ""
    pressure: str = ""
    visibility: str = ""
    aqi: int | None = None
    aqi_text: str = ""
    daily: list[DailyForecast] = field(default_factory=list)
    indices: dict[str, dict[str, str]] = field(default_factory=dict)
    alerts: list[dict[str, Any]] = field(default_factory=list)
    updated: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    # 便捷属性 ---------------------------------------------------------
    @property
    def today(self) -> DailyForecast | None:
        return self.daily[0] if self.daily else None

    @property
    def city(self) -> str:
        return self.location.name

    @property
    def today_max(self) -> float | None:
        return self.today.temp_max if self.today else None

    @property
    def today_min(self) -> float | None:
        return self.today.temp_min if self.today else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "location": self.location.to_dict(),
            "source": self.source,
            "source_label": self.source_label,
            "current_text": self.current_text,
            "current_temp": self.current_temp,
            "feels_like": self.feels_like,
            "humidity": self.humidity,
            "wind_dir": self.wind_dir,
            "wind_scale": self.wind_scale,
            "pressure": self.pressure,
            "visibility": self.visibility,
            "aqi": self.aqi,
            "aqi_text": self.aqi_text,
            "today_max": self.today_max,
            "today_min": self.today_min,
            "daily": [d.to_dict() for d in self.daily],
            "indices": self.indices,
            "alerts": self.alerts,
            "updated": self.updated,
        }

    def to_prompt_text(
        self,
        max_days: int = 3,
        *,
        show_alerts: bool = True,
        show_air: bool = True,
        aspects: set[str] | None = None,
        show_indices: bool = True,
    ) -> str:
        parts: list[str] = []
        parts.append(f"城市：{self.location.name}")
        if (
            self.location.province
            and self.location.province != self.location.name
        ):
            parts.append(f"省份：{self.location.province}")
        now_bits = []
        if self.current_text:
            now_bits.append(self.current_text)
        if self.current_temp is not None:
            now_bits.append(f"{format_temp(self.current_temp)}℃")
        if self.feels_like is not None and (
            self.current_temp is None
            or abs(self.feels_like - self.current_temp) >= 2
        ):
            now_bits.append(f"体感{format_temp(self.feels_like)}℃")
        if self.humidity is not None:
            now_bits.append(f"湿度{self.humidity}%")
        if self.wind_dir or self.wind_scale:
            now_bits.append(f"{self.wind_dir}{self.wind_scale}".strip())
        if show_air and self.aqi is not None:
            parts.append(
                "空气："
                + (f"{self.aqi_text}({self.aqi})" if self.aqi_text else str(self.aqi))
            )
        if now_bits:
            parts.append("现在：" + "，".join(now_bits))

        for idx, d in enumerate(self.daily[: max(1, min(max_days, 7))]):
            bits = [f"{d.label or d.date}"]
            text = d.text or ""
            if d.text_night and d.text_night != d.text:
                text = f"{text}转{d.text_night}"
            if text:
                bits.append(text)
            if d.temp_min is not None and d.temp_max is not None:
                bits.append(f"{format_temp(d.temp_min)}~{format_temp(d.temp_max)}℃")
            if d.precip_prob is not None:
                bits.append(f"降水{d.precip_prob}%")
            if d.wind_dir or d.wind_scale:
                bits.append(f"{d.wind_dir}{d.wind_scale}".strip())
            parts.append("，".join(bits))

        if show_indices and self.indices:
            picked = pick_indices(self, aspects)
            if picked:
                parts.append(
                    "生活指数："
                    + "；".join(
                        f"{name}{hint}"
                        + (f"（{desc}）" if desc and len(desc) <= 40 else "")
                        for name, hint, desc in picked
                    )
                )

        if show_alerts and self.alerts:
            titles = "；".join(str(a.get("title") or "") for a in self.alerts[:2])
            if titles.strip():
                parts.append(f"预警：{titles}")
        return "\n".join(parts)

    def summary_line(self) -> str:
        bits = [self.location.display]
        if self.current_text:
            bits.append(self.current_text)
        if self.current_temp is not None:
            bits.append(f"{format_temp(self.current_temp)}℃")
        if self.today_min is not None and self.today_max is not None:
            bits.append(
                f"今天{format_temp(self.today_min)}~{format_temp(self.today_max)}℃"
            )
        return "  ".join(bits)


def _to_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _sane_temp(value: Any) -> float | None:
    """过滤数据源里的占位值（中国天气网偶尔返回 999）。"""
    temp = _to_float(value)
    if temp is None or temp < -90 or temp > 60:
        return None
    return temp


def _to_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def format_temp(value: float | None) -> str:
    if value is None:
        return "--"
    num = round(float(value))
    if float(num) == value:
        return str(num)
    return f"{value:.1f}"


# ---------------------------------------------------------------------------
# 天气代码 / 风力 / AQI
# ---------------------------------------------------------------------------


CN_WEATHER_CODES: dict[str, str] = {
    "00": "晴",
    "01": "多云",
    "02": "阴",
    "03": "阵雨",
    "04": "雷阵雨",
    "05": "雷阵雨伴冰雹",
    "06": "雨夹雪",
    "07": "小雨",
    "08": "中雨",
    "09": "大雨",
    "10": "暴雨",
    "11": "大暴雨",
    "12": "特大暴雨",
    "13": "阵雪",
    "14": "小雪",
    "15": "中雪",
    "16": "大雪",
    "17": "暴雪",
    "18": "雾",
    "19": "冻雨",
    "20": "沙尘暴",
    "21": "小到中雨",
    "22": "中到大雨",
    "23": "大到暴雨",
    "24": "暴雨到大暴雨",
    "25": "大暴雨到特大暴雨",
    "26": "小到中雪",
    "27": "中到大雪",
    "28": "大到暴雪",
    "29": "浮尘",
    "30": "扬沙",
    "31": "强沙尘暴",
    "32": "飓风",
    "49": "强浓雾",
    "53": "霾",
    "54": "中度霾",
    "55": "重度霾",
    "56": "严重霾",
    "99": "无",
}

WMO_WEATHER_CODES: dict[int, str] = {
    0: "晴",
    1: "基本晴",
    2: "多云",
    3: "阴",
    45: "雾",
    48: "雾凇",
    51: "毛毛雨",
    53: "毛毛雨",
    55: "强毛毛雨",
    56: "冻毛毛雨",
    57: "强冻毛毛雨",
    61: "小雨",
    63: "中雨",
    65: "大雨",
    66: "冻雨",
    67: "强冻雨",
    71: "小雪",
    73: "中雪",
    75: "大雪",
    77: "米雪",
    80: "阵雨",
    81: "强阵雨",
    82: "暴雨",
    85: "阵雪",
    86: "强阵雪",
    95: "雷阵雨",
    96: "雷阵雨伴冰雹",
    99: "强雷阵雨伴冰雹",
}

WWO_WEATHER_CODES: dict[int, str] = {
    113: "晴",
    116: "多云",
    119: "阴",
    122: "阴",
    143: "薄雾",
    176: "局部有雨",
    179: "局部有雪",
    182: "局部雨夹雪",
    185: "局部冻毛毛雨",
    200: "雷阵雨",
    227: "吹雪",
    230: "暴风雪",
    248: "雾",
    260: "冻雾",
    263: "零星毛毛雨",
    266: "毛毛雨",
    281: "冻毛毛雨",
    284: "强冻毛毛雨",
    293: "零星小雨",
    296: "小雨",
    299: "间歇中雨",
    302: "中雨",
    305: "间歇大雨",
    308: "大雨",
    311: "小冻雨",
    314: "中到大冻雨",
    317: "小雨夹雪",
    320: "中到大雨夹雪",
    323: "局部小雪",
    326: "小雪",
    329: "局部中雪",
    332: "中雪",
    335: "局部大雪",
    338: "大雪",
    350: "冰粒",
    353: "小阵雨",
    356: "中到大阵雨",
    359: "暴雨",
    362: "小雨夹雪阵",
    365: "中到大雨夹雪阵",
    368: "小阵雪",
    371: "中到大阵雪",
    374: "小冰粒阵",
    377: "中到大冰粒阵",
    386: "局部雷阵雨",
    389: "强雷阵雨",
    392: "局部雷阵雪",
    395: "强雷阵雪",
}

WIND_DIRS_CN = ["北风", "东北风", "东风", "东南风", "南风", "西南风", "西风", "西北风"]

COMPASS_16_TO_CN = {
    "N": "北风",
    "NNE": "东北风",
    "NE": "东北风",
    "ENE": "东北风",
    "E": "东风",
    "ESE": "东南风",
    "SE": "东南风",
    "SSE": "东南风",
    "S": "南风",
    "SSW": "西南风",
    "SW": "西南风",
    "WSW": "西南风",
    "W": "西风",
    "WNW": "西北风",
    "NW": "西北风",
    "NNW": "西北风",
}


def degrees_to_cn(deg: float | None) -> str:
    if deg is None:
        return ""
    idx = int((float(deg) % 360) / 45.0 + 0.5) % 8
    return WIND_DIRS_CN[idx]


def kmh_to_scale(kmh: float | None) -> str:
    if kmh is None:
        return ""
    thresholds = [
        (1, 0),
        (5, 1),
        (11, 2),
        (19, 3),
        (28, 4),
        (38, 5),
        (49, 6),
        (61, 7),
        (74, 8),
        (88, 9),
        (102, 10),
        (117, 11),
    ]
    for limit, scale in thresholds:
        if kmh < limit:
            return f"{scale}级"
    return "12级"


def aqi_to_text(aqi: int | None) -> str:
    if aqi is None:
        return ""
    if aqi <= 50:
        return "优"
    if aqi <= 100:
        return "良"
    if aqi <= 150:
        return "轻度污染"
    if aqi <= 200:
        return "中度污染"
    if aqi <= 300:
        return "重度污染"
    return "严重污染"


def _is_rainy(text: str) -> bool:
    return any(k in (text or "") for k in ("雨", "雪", "雷", "冰雹", "阵雨", "冻"))


# ---------------------------------------------------------------------------
# 数据源：中国天气网
# ---------------------------------------------------------------------------


class CNWeatherProvider:
    id = "cnweather"
    label = "中国天气网"

    SEARCH_URL = "http://toy1.weather.com.cn/search?cityname={name}&_={ts}"
    INDEX_URL = "http://d1.weather.com.cn/weather_index/{code}.html?_{ts}"
    REFERER = "http://www.weather.com.cn/"

    async def search(
        self, name: str, *, timeout: float, retries: int
    ) -> list[dict[str, str]]:
        # 带时间戳绕过 CDN 的空结果缓存（实测「杭州」会被缓存成 ()）
        url = self.SEARCH_URL.format(
            name=urllib.parse.quote(name), ts=int(time.time() * 1000)
        )
        raw, charset = await asyncio.to_thread(
            _http_get,
            url,
            timeout=timeout,
            retries=retries,
            headers={"Referer": self.REFERER},
        )
        text = _decode(raw, charset)
        results: list[dict[str, str]] = []
        for match in re.finditer(r'"ref":"([^"]+)"', text):
            fields = match.group(1).split("~")
            if len(fields) < 3:
                continue
            results.append(
                {
                    "code": fields[0].strip(),
                    "name": fields[2].strip(),
                    "province": (fields[9] if len(fields) > 9 else "").strip(),
                }
            )
        return results

    async def fetch(
        self, loc: Location, *, days: int, timeout: float, retries: int
    ) -> WeatherData:
        code = (loc.cn_code or "").strip()
        if not code:
            found = await self.search(loc.name, timeout=timeout, retries=retries)
            best = _pick_cn_result(found, loc.name, loc.province)
            if best is None:
                raise ProviderError(f"中国天气网找不到「{loc.name}」")
            code = best["code"]
            loc = Location(
                name=best.get("name") or loc.name,
                province=best.get("province") or loc.province,
                lat=loc.lat,
                lon=loc.lon,
                cn_code=code,
            )

        url = self.INDEX_URL.format(
            code=urllib.parse.quote(code), ts=int(time.time() * 1000)
        )
        raw, charset = await asyncio.to_thread(
            _http_get,
            url,
            timeout=timeout,
            retries=retries,
            headers={"Referer": self.REFERER},
        )
        text = _decode(raw, charset)

        sk = _extract_js_object(text, "dataSK") or {}
        if not sk:
            raise ProviderError("中国天气网返回内容缺少 dataSK")

        city_dz = _extract_js_object(text, "cityDZ") or {}
        fc_obj = _extract_js_object(text, "fc") or {}
        alarm = _extract_js_object(text, "alarmDZ") or {}
        zs_obj = _extract_js_object(text, "dataZS") or {}

        today_info = (city_dz.get("weatherinfo") or {}) if isinstance(city_dz, dict) else {}
        fc_list = fc_obj.get("f") if isinstance(fc_obj, dict) else None
        if not isinstance(fc_list, list):
            fc_list = []

        # 当前
        cur_text = str(sk.get("weather") or "").strip()
        cur_temp = _sane_temp(sk.get("temp"))
        humidity = _to_int(str(sk.get("SD") or sk.get("sd") or "").replace("%", ""))
        aqi = _to_int(sk.get("aqi"))
        wind_dir = str(sk.get("WD") or "").strip()
        wind_scale = str(sk.get("WS") or "").strip()

        # 今日高低温：优先 fc 第一天
        daily: list[DailyForecast] = []
        year = _year_from_text(str(sk.get("date") or ""))
        for item in fc_list[: max(1, min(days, 7))]:
            if not isinstance(item, dict):
                continue
            date_iso = _parse_cn_fc_date(str(item.get("fi") or ""), year)
            day_text = CN_WEATHER_CODES.get(str(item.get("fa") or ""), "")
            night_text = CN_WEATHER_CODES.get(str(item.get("fb") or ""), "")
            daily.append(
                DailyForecast(
                    date=date_iso,
                    label="",
                    text=day_text,
                    text_night=night_text,
                    temp_max=_sane_temp(item.get("fc")),
                    temp_min=_sane_temp(item.get("fd")),
                    wind_dir=str(item.get("fe") or "").strip(),
                    wind_scale=str(item.get("fg") or "").strip(),
                )
            )
        _fill_labels(daily)

        # 实时天气代码比 fc 更准
        code_cur = str(sk.get("weathercode") or "")
        if code_cur[:2] in CN_WEATHER_CODES and cur_text:
            # dataSK 的 weather 已经是中文，无需替换
            pass

        data = WeatherData(
            location=loc,
            source=self.id,
            source_label=self.label,
            current_text=cur_text,
            current_temp=cur_temp,
            feels_like=None,
            humidity=humidity,
            wind_dir=wind_dir,
            wind_scale=wind_scale,
            pressure=str(sk.get("qy") or "").strip(),
            visibility=str(sk.get("njd") or "").strip(),
            aqi=aqi,
            aqi_text=aqi_to_text(aqi),
            daily=daily,
            indices=_parse_cn_indices(zs_obj),
            alerts=_parse_cn_alerts(alarm),
            updated=f"{sk.get('date') or ''} {sk.get('time') or ''}".strip(),
            raw={"dataSK": sk, "cityDZ": today_info},
        )
        if not data.current_text and daily:
            data.current_text = daily[0].text
        return data


def _extract_js_object(text: str, var_name: str) -> dict[str, Any] | None:
    """从 `var xxx = {...};` 里抠出 JSON 对象（允许后面跟其他内容）。"""
    pattern = re.compile(r"var\s+" + re.escape(var_name) + r"\s*=\s*", re.S)
    match = pattern.search(text)
    if not match:
        return None
    start = match.end()
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                chunk = text[start : idx + 1]
                try:
                    return json.loads(chunk)
                except json.JSONDecodeError:
                    return None
    return None


def _year_from_text(text: str) -> int:
    match = re.search(r"(20\d{2})", text)
    if match:
        return int(match.group(1))
    return time.localtime().tm_year


def _parse_cn_fc_date(raw: str, year: int) -> str:
    match = re.match(r"(\d{1,2})/(\d{1,2})", raw.strip())
    if not match:
        return ""
    month, day = int(match.group(1)), int(match.group(2))
    try:
        return f"{year:04d}-{month:02d}-{day:02d}"
    except ValueError:
        return ""


def _parse_cn_alerts(alarm: dict[str, Any]) -> list[dict[str, Any]]:
    items = alarm.get("w") if isinstance(alarm, dict) else None
    if not isinstance(items, list):
        return []
    result: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        title_parts = [
            str(item.get("w1") or "").strip(),
            str(item.get("w2") or "").strip(),
            str(item.get("w5") or "").strip(),
            str(item.get("w7") or "").strip(),
        ]
        title = "".join(part for part in title_parts if part)
        if title and "预警" not in title:
            title += "预警"
        result.append(
            {
                "title": title,
                "type": str(item.get("w5") or "").strip(),
                "level": str(item.get("w7") or "").strip(),
                "time": str(item.get("w8") or "").strip(),
                "text": str(item.get("w9") or "").strip(),
            }
        )
    return result


def _parse_cn_indices(zs_obj: dict[str, Any]) -> dict[str, dict[str, str]]:
    zs = zs_obj.get("zs") if isinstance(zs_obj, dict) else None
    if not isinstance(zs, dict):
        return {}
    result: dict[str, dict[str, str]] = {}
    for key, value in zs.items():
        if not key.endswith("_name"):
            continue
        prefix = key[: -len("_name")]
        name = str(value or "").strip()
        if not name:
            continue
        result[prefix] = {
            "name": name,
            "hint": str(zs.get(f"{prefix}_hint") or "").strip(),
            "desc": str(zs.get(f"{prefix}_des_s") or "").strip(),
        }
    return result


def _fill_labels(daily: list[DailyForecast]) -> None:
    weekday_cn = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    today = _dt.date.today()
    for d in daily:
        if not d.date:
            d.label = d.label or ""
            continue
        label = ""
        try:
            dt = _dt.date.fromisoformat(d.date)
            delta = (dt - today).days
            if delta == 0:
                label = "今天"
            elif delta == 1:
                label = "明天"
            elif delta == 2:
                label = "后天"
            else:
                label = weekday_cn[dt.weekday()]
        except (ValueError, OverflowError):
            label = ""
        d.label = label or d.date


# ---------------------------------------------------------------------------
# 数据源：Open-Meteo
# ---------------------------------------------------------------------------


class OpenMeteoProvider:
    id = "open_meteo"
    label = "Open-Meteo"

    GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"
    FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
    AIR_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

    async def geocode(
        self, name: str, *, timeout: float, retries: int
    ) -> dict[str, Any] | None:
        params = urllib.parse.urlencode(
            {"name": name, "count": 5, "language": "zh", "format": "json"}
        )
        payload = await asyncio.to_thread(
            _get_json, f"{self.GEO_URL}?{params}", timeout=timeout, retries=retries
        )
        results = payload.get("results") if isinstance(payload, dict) else None
        if not results:
            return None
        first = results[0]
        admin1 = str(first.get("admin1") or "").strip()
        return {
            "name": str(first.get("name") or name).strip(),
            "province": admin1,
            "lat": _to_float(first.get("latitude")),
            "lon": _to_float(first.get("longitude")),
            "country": str(first.get("country_code") or "").strip(),
        }

    async def fetch(
        self,
        loc: Location,
        *,
        days: int,
        timeout: float,
        retries: int,
        include_air: bool = True,
    ) -> WeatherData:
        if loc.lat is None or loc.lon is None:
            raise ProviderError("Open-Meteo 需要经纬度")

        params = urllib.parse.urlencode(
            {
                "latitude": loc.lat,
                "longitude": loc.lon,
                "current": ",".join(
                    [
                        "temperature_2m",
                        "relative_humidity_2m",
                        "apparent_temperature",
                        "weather_code",
                        "wind_speed_10m",
                        "wind_direction_10m",
                        "pressure_msl",
                    ]
                ),
                "daily": ",".join(
                    [
                        "weather_code",
                        "temperature_2m_max",
                        "temperature_2m_min",
                        "precipitation_probability_max",
                        "wind_speed_10m_max",
                        "wind_direction_10m_dominant",
                    ]
                ),
                "timezone": "auto",
                "forecast_days": max(1, min(int(days), 7)),
            }
        )
        payload = await asyncio.to_thread(
            _get_json, f"{self.FORECAST_URL}?{params}", timeout=timeout, retries=retries
        )
        current = payload.get("current") or {}
        daily_obj = payload.get("daily") or {}
        times = daily_obj.get("time") or []
        code_list = daily_obj.get("weather_code") or []
        max_list = daily_obj.get("temperature_2m_max") or []
        min_list = daily_obj.get("temperature_2m_min") or []
        prob_list = daily_obj.get("precipitation_probability_max") or []
        wind_speed_list = daily_obj.get("wind_speed_10m_max") or []
        wind_dir_list = daily_obj.get("wind_direction_10m_dominant") or []

        daily: list[DailyForecast] = []
        for idx, date_str in enumerate(times):
            code = _to_int(code_list[idx]) if idx < len(code_list) else None
            text = WMO_WEATHER_CODES.get(code, "") if code is not None else ""
            daily.append(
                DailyForecast(
                    date=str(date_str),
                    label="",
                    text=text,
                    text_night="",
                    temp_max=_sane_temp(max_list[idx]) if idx < len(max_list) else None,
                    temp_min=_sane_temp(min_list[idx]) if idx < len(min_list) else None,
                    precip_prob=(
                        _to_int(prob_list[idx]) if idx < len(prob_list) else None
                    ),
                    wind_dir=(
                        degrees_to_cn(_to_float(wind_dir_list[idx]))
                        if idx < len(wind_dir_list)
                        else ""
                    ),
                    wind_scale=(
                        kmh_to_scale(_to_float(wind_speed_list[idx]))
                        if idx < len(wind_speed_list)
                        else ""
                    ),
                )
            )
        _fill_labels(daily)

        cur_code = _to_int(current.get("weather_code"))
        cur_text = WMO_WEATHER_CODES.get(cur_code, "") if cur_code is not None else ""
        wind_speed = _to_float(current.get("wind_speed_10m"))
        aqi, aqi_text = (None, "")
        if include_air:
            try:
                aqi, aqi_text = await self._fetch_air(loc, timeout=min(timeout, 5), retries=0)
            except Exception:  # noqa: BLE001 - 空气质量是附加信息
                aqi, aqi_text = (None, "")

        data = WeatherData(
            location=loc,
            source=self.id,
            source_label=self.label,
            current_text=cur_text,
            current_temp=_sane_temp(current.get("temperature_2m")),
            feels_like=_sane_temp(current.get("apparent_temperature")),
            humidity=_to_int(current.get("relative_humidity_2m")),
            wind_dir=degrees_to_cn(_to_float(current.get("wind_direction_10m"))),
            wind_scale=kmh_to_scale(wind_speed),
            pressure=(
                f"{format_temp(_to_float(current.get('pressure_msl')))}hPa"
                if current.get("pressure_msl") is not None
                else ""
            ),
            aqi=aqi,
            aqi_text=aqi_text,
            daily=daily,
            updated=str(current.get("time") or payload.get("timezone") or ""),
            raw={"current": current, "daily": daily_obj},
        )
        if not data.current_text and daily:
            data.current_text = daily[0].text
        return data

    async def fetch_hourly(
        self, loc: Location, *, timeout: float, retries: int, hours: int = 12
    ) -> list[HourlyPoint]:
        if loc.lat is None or loc.lon is None:
            raise ProviderError("Open-Meteo 小时级预报需要经纬度")
        params = urllib.parse.urlencode(
            {
                "latitude": loc.lat,
                "longitude": loc.lon,
                "hourly": "precipitation_probability,precipitation,weather_code,temperature_2m",
                "forecast_hours": max(1, min(int(hours), 24)),
                "timezone": "auto",
            }
        )
        payload = await asyncio.to_thread(
            _get_json, f"{self.FORECAST_URL}?{params}", timeout=timeout, retries=retries
        )
        hourly = payload.get("hourly") or {}
        times = hourly.get("time") or []
        probs = hourly.get("precipitation_probability") or []
        precips = hourly.get("precipitation") or []
        codes = hourly.get("weather_code") or []
        temps = hourly.get("temperature_2m") or []
        points: list[HourlyPoint] = []
        for idx, raw_time in enumerate(times):
            raw_text = str(raw_time)
            clock = raw_text.split("T")[-1][:5] if "T" in raw_text else raw_text[-5:]
            code = _to_int(codes[idx]) if idx < len(codes) else None
            points.append(
                HourlyPoint(
                    time=clock,
                    temp=_sane_temp(temps[idx]) if idx < len(temps) else None,
                    text=WMO_WEATHER_CODES.get(code, "") if code is not None else "",
                    precip_prob=_to_int(probs[idx]) if idx < len(probs) else None,
                    precip_mm=_to_float(precips[idx]) if idx < len(precips) else None,
                )
            )
        return points

    async def _fetch_air(
        self, loc: Location, *, timeout: float, retries: int
    ) -> tuple[int | None, str]:
        params = urllib.parse.urlencode(
            {
                "latitude": loc.lat,
                "longitude": loc.lon,
                "current": "us_aqi,pm2_5",
                "timezone": "auto",
            }
        )
        payload = await asyncio.to_thread(
            _get_json, f"{self.AIR_URL}?{params}", timeout=timeout, retries=retries
        )
        current = payload.get("current") or {}
        aqi = _to_int(current.get("us_aqi"))
        return aqi, aqi_to_text(aqi)


# ---------------------------------------------------------------------------
# 数据源：wttr.in
# ---------------------------------------------------------------------------


class WttrProvider:
    id = "wttr"
    label = "wttr.in"

    BASE = "https://wttr.in/{query}?format=j1&lang=zh"

    async def fetch(
        self, loc: Location, *, days: int, timeout: float, retries: int
    ) -> WeatherData:
        if loc.lat is not None and loc.lon is not None:
            query = f"{loc.lat:.4f},{loc.lon:.4f}"
        else:
            query = loc.name
        url = self.BASE.format(query=urllib.parse.quote(query, safe=","))
        payload = await asyncio.to_thread(
            _get_json, url, timeout=timeout, retries=retries
        )
        if not isinstance(payload, dict):
            raise ProviderError("wttr.in 返回格式异常")

        current_list = payload.get("current_condition") or []
        if not current_list:
            raise ProviderError("wttr.in 没有返回实况")
        cur = current_list[0]
        weather_list = payload.get("weather") or []

        cur_code = _to_int(cur.get("weatherCode"))
        cur_text = WWO_WEATHER_CODES.get(cur_code, "") or _desc_cn(cur)
        wind_kmph = _to_float(cur.get("windspeedKmph"))
        wind_dir = COMPASS_16_TO_CN.get(str(cur.get("winddir16Point") or "").upper(), "")
        aqi = _to_int(cur.get("airqualityindex"))

        daily: list[DailyForecast] = []
        for idx, item in enumerate(weather_list[: max(1, min(days, 7))]):
            if not isinstance(item, dict):
                continue
            hourly = item.get("hourly") or []
            midday = hourly[4] if len(hourly) > 4 else (hourly[0] if hourly else {})
            code = _to_int(midday.get("weatherCode"))
            text = WWO_WEATHER_CODES.get(code, "") if code is not None else ""
            if not text:
                text = _desc_cn(midday)
            chances = [
                _to_int(h.get("chanceofrain")) or 0
                for h in hourly
                if isinstance(h, dict)
            ]
            daily.append(
                DailyForecast(
                    date=str(item.get("date") or ""),
                    label="",
                    text=text,
                    text_night="",
                    temp_max=_sane_temp(item.get("maxtempC")),
                    temp_min=_sane_temp(item.get("mintempC")),
                    precip_prob=max(chances) if chances else None,
                    wind_dir="",
                    wind_scale="",
                )
            )
        _fill_labels(daily)

        data = WeatherData(
            location=loc,
            source=self.id,
            source_label=self.label,
            current_text=cur_text,
            current_temp=_sane_temp(cur.get("temp_C")),
            feels_like=_sane_temp(cur.get("FeelsLikeC")),
            humidity=_to_int(cur.get("humidity")),
            wind_dir=wind_dir,
            wind_scale=kmh_to_scale(wind_kmph),
            pressure=f"{cur.get('pressure')}hPa" if cur.get("pressure") else "",
            visibility=f"{cur.get('visibility')}km" if cur.get("visibility") else "",
            aqi=aqi,
            aqi_text=aqi_to_text(aqi) if aqi else "",
            daily=daily,
            updated=str(cur.get("observation_time") or ""),
            raw={"current": cur},
        )
        if not data.current_text and daily:
            data.current_text = daily[0].text
        return data

    async def fetch_hourly(
        self, loc: Location, *, timeout: float, retries: int, hours: int = 12
    ) -> list[HourlyPoint]:
        if loc.lat is not None and loc.lon is not None:
            query = f"{loc.lat:.4f},{loc.lon:.4f}"
        else:
            query = loc.name
        url = self.BASE.format(query=urllib.parse.quote(query, safe=","))
        payload = await asyncio.to_thread(
            _get_json, url, timeout=timeout, retries=retries
        )
        weather_list = payload.get("weather") if isinstance(payload, dict) else None
        if not weather_list:
            raise ProviderError("wttr.in 没有小时级数据")
        now_hour = time.localtime().tm_hour
        points: list[HourlyPoint] = []
        for day in weather_list[:2]:
            for item in day.get("hourly") or []:
                raw_time = str(item.get("time") or "0").zfill(4)
                hour = int(raw_time[:-2] or 0)
                if not points and hour < now_hour:
                    continue
                code = _to_int(item.get("weatherCode"))
                points.append(
                    HourlyPoint(
                        time=f"{hour:02d}:00",
                        temp=_sane_temp(item.get("tempC")),
                        text=WWO_WEATHER_CODES.get(code, "") if code is not None else "",
                        precip_prob=_to_int(item.get("chanceofrain")),
                        precip_mm=_to_float(item.get("precipMM")),
                    )
                )
                if len(points) >= max(1, min(int(hours), 24)):
                    return points
        return points


def _desc_cn(item: dict[str, Any]) -> str:
    for key in ("lang_zh", "weatherDesc"):
        values = item.get(key)
        if isinstance(values, list) and values:
            value = str(values[0].get("value") or "").strip()
            if value and not re.search(r"[A-Za-z]{3,}", value):
                return value
    # 英文描述兜底
    values = item.get("weatherDesc")
    if isinstance(values, list) and values:
        english = str(values[0].get("value") or "").strip().lower()
        mapping = {
            "sunny": "晴",
            "clear": "晴",
            "partly cloudy": "多云",
            "cloudy": "多云",
            "overcast": "阴",
            "mist": "薄雾",
            "fog": "雾",
            "haze": "霾",
            "smoky haze": "霾",
            "light rain": "小雨",
            "moderate rain": "中雨",
            "heavy rain": "大雨",
            "light snow": "小雪",
            "moderate snow": "中雪",
            "heavy snow": "大雪",
            "thunderstorm": "雷阵雨",
            "patchy rain possible": "局部有雨",
            "patchy snow possible": "局部有雪",
        }
        for key, value in mapping.items():
            if key in english:
                return value
        return english
    return ""


# ---------------------------------------------------------------------------
# 地点解析
# ---------------------------------------------------------------------------


LOCATION_STOPWORDS = {
    "今天",
    "明天",
    "后天",
    "现在",
    "最近",
    "这两天",
    "这几天",
    "这周",
    "本周末",
    "周末",
    "下周",
    "外面",
    "这里",
    "那边",
    "这边",
    "我们",
    "你们",
    "他们",
    "我这边",
    "你那边",
    "家里",
    "公司",
    "学校",
    "路上",
    "早上",
    "中午",
    "晚上",
    "白天",
    "夜里",
    "今天早上",
    "今天晚上",
}

# 省级名称：全称 -> 简称。用于「山东省临沂市」「浙江杭州」这类输入。
PROVINCE_FULL_NAMES: dict[str, str] = {
    "河北省": "河北",
    "山西省": "山西",
    "辽宁省": "辽宁",
    "吉林省": "吉林",
    "黑龙江省": "黑龙江",
    "江苏省": "江苏",
    "浙江省": "浙江",
    "安徽省": "安徽",
    "福建省": "福建",
    "江西省": "江西",
    "山东省": "山东",
    "河南省": "河南",
    "湖北省": "湖北",
    "湖南省": "湖南",
    "广东省": "广东",
    "海南省": "海南",
    "四川省": "四川",
    "贵州省": "贵州",
    "云南省": "云南",
    "陕西省": "陕西",
    "甘肃省": "甘肃",
    "青海省": "青海",
    "台湾省": "台湾",
    "内蒙古自治区": "内蒙古",
    "广西壮族自治区": "广西",
    "西藏自治区": "西藏",
    "宁夏回族自治区": "宁夏",
    "新疆维吾尔自治区": "新疆",
    "香港特别行政区": "香港",
    "澳门特别行政区": "澳门",
    "北京市": "北京",
    "天津市": "天津",
    "上海市": "上海",
    "重庆市": "重庆",
}

PROVINCE_SHORT_NAMES = {
    "河北",
    "山西",
    "辽宁",
    "吉林",
    "黑龙江",
    "江苏",
    "浙江",
    "安徽",
    "福建",
    "江西",
    "山东",
    "河南",
    "湖北",
    "湖南",
    "广东",
    "海南",
    "四川",
    "贵州",
    "云南",
    "陕西",
    "甘肃",
    "青海",
    "台湾",
    "内蒙古",
    "广西",
    "西藏",
    "宁夏",
    "新疆",
    "香港",
    "澳门",
    "北京",
    "天津",
    "上海",
    "重庆",
}

MUNICIPALITIES = {"北京", "天津", "上海", "重庆", "香港", "澳门"}
DISTRICT_SUFFIXES = ("区", "县", "旗")


def _strip_admin_suffixes(name: str) -> str:
    result = (name or "").strip()
    while result.endswith(("省", "市", "区", "县", "旗", "盟")):
        result = result[:-1].strip()
    return result


def _split_province(name: str) -> tuple[str, str]:
    """返回 (省简称, 去掉省后的部分)。"""
    for full in sorted(PROVINCE_FULL_NAMES, key=len, reverse=True):
        if name.startswith(full):
            return PROVINCE_FULL_NAMES[full], name[len(full) :].strip()
    for short in sorted(PROVINCE_SHORT_NAMES, key=len, reverse=True):
        if not name.startswith(short) or len(name) <= len(short):
            continue
        # 吉林市是城市，不是吉林省
        if short == "吉林" and name.startswith("吉林市"):
            continue
        return short, name[len(short) :].strip()
    return "", name


def _split_city_district(rest: str) -> tuple[str, str]:
    """把「临沂市兰山区」拆成 (临沂, 兰山区)。"""
    if "市" in rest:
        index = rest.find("市")
        return rest[:index].strip(), rest[index + 1 :].strip()
    for suffix in DISTRICT_SUFFIXES:
        index = rest.find(suffix)
        if index >= 2:
            return rest[:index].strip(), rest[index:].strip()
    return rest.strip(), ""


def chinese_location_variants(name: str) -> list[str]:
    """从「山东省临沂市兰山区」这类输入里生成按优先级排列的城市候选。"""
    raw = _normalize_location_name(name)
    if not raw:
        return []
    # 先按原始输入匹配省级全称（内蒙古自治区/香港特别行政区等含「区」，
    # 不能先去掉后缀），匹配不到再去后缀重试
    province, rest = _split_province(raw)
    cleaned = raw
    if not province:
        stripped = _strip_admin_suffixes(raw)
        province, rest = _split_province(stripped)
        if province:
            cleaned = stripped
        else:
            rest = raw

    variants: list[str] = []

    def add(value: str) -> None:
        value = _strip_admin_suffixes(value)
        if value and value not in variants and len(value) <= 20:
            variants.append(value)

    def add_prefixes(value: str) -> None:
        """「杭州余杭」这种没有“市”字的，按 4/3/2 字前缀试城市。"""
        base = _strip_admin_suffixes(value)
        for size in (4, 3, 2):
            if len(base) > size:
                add(base[:size])
        add(base)

    def add_tails(value: str) -> None:
        """「杭州余杭」先试结尾 3/2 字，命中区县就更精确。"""
        base = _strip_admin_suffixes(value)
        for size in (3, 2):
            if len(base) >= size:
                add(base[-size:])

    # 只说了省名，不拿省去查
    if (
        not province
        and rest in PROVINCE_SHORT_NAMES
        and rest not in MUNICIPALITIES
        and rest != "吉林"
    ):
        return []

    if province in MUNICIPALITIES:
        if rest:
            # 「北京朝阳区」：先试区县，再退回城市
            add_tails(rest)
            add_prefixes(rest)
            add(province)
        else:
            add(province)
    elif province and not rest:
        return []
    else:
        body = rest or cleaned
        if "市" in body:
            city, district = body.split("市", 1)
            if district:
                add(district)
                add_prefixes(district)
            add(city)
        else:
            add_tails(body)
            add_prefixes(body)
        add(rest)

    add(cleaned)
    # 再对每个候选逐级去掉「市/区/县」后缀，兜底旧逻辑
    for base in list(variants):
        trimmed = base
        while True:
            new = _strip_admin_suffixes(trimmed)
            if new == trimmed:
                break
            add(new)
            trimmed = new
    return variants


def _normalize_location_name(raw: str) -> str:
    name = (raw or "").strip()
    name = re.sub(r"[\s，,。.!！?？~～、:：;；\"'“”‘’()（）\[\]【】]+", "", name)
    return name


NON_PLACE_WORDS = {
    "上班",
    "下班",
    "加班",
    "睡觉",
    "吃饭",
    "洗澡",
    "上课",
    "开会",
    "出门",
    "在家",
    "家里",
    "宿舍",
    "单位",
    "公司",
    "学校",
    "路上",
    "地铁",
    "公交",
    "医院",
    "商场",
    "超市",
    "外面",
    "屋里",
    "打游戏",
    "看电影",
}


def looks_like_location(text: str) -> bool:
    name = _normalize_location_name(text)
    if not name or len(name) > 20:
        return False
    if name in LOCATION_STOPWORDS or name in NON_PLACE_WORDS:
        return False
    if re.search(r"天气|气温|温度|下雨|下雪|带伞|几度|多少度|冷|热|穿", name):
        return False
    if re.search(r"[A-Za-z]{2,}", name) and len(name) > 2:
        # 允许 "Shanghai" 这种纯英文地名
        return bool(re.fullmatch(r"[A-Za-z\s\-']+", name))
    return bool(re.fullmatch(r"[\u4e00-\u9fa5A-Za-z·\-\s]{2,20}", name))


def clean_location_answer(text: str) -> str:
    """把「我在杭州呀」「杭州」这类回答洗成地名。"""
    name = _normalize_location_name(text)
    prefixes = [
        "我住在",
        "我住",
        "我在",
        "我这里是",
        "我这边是",
        "这边是",
        "这里是",
        "坐标",
        "位置",
        "在",
    ]
    for prefix in prefixes:
        if name.startswith(prefix) and len(name) > len(prefix) + 1:
            name = name[len(prefix) :]
            break
    suffixes = ["这边", "那儿", "那里", "附近", "本地", "市里", "市区"]
    for suffix in suffixes:
        if name.endswith(suffix) and len(name) > len(suffix) + 1:
            name = name[: -len(suffix)]
            break
    name = _TIME_PREFIX_RE.sub("", name).strip()
    name = re.sub(r"(呀|啦|呢|吧|哦|嘛|啊|哈|哟|喔)+$", "", name).strip()
    return name


def strip_weather_suffix(text: str) -> str:
    """去掉「山东省临沂市兰山区天气怎么样」里的问句尾巴，只留地名。"""
    raw = (text or "").strip()
    raw = re.sub(
        r"(?:今天|明天|后天|现在|最近|这周|本周|下周|周末)?"
        r"(?:的)?(?:会|要|会不会|可能)?"
        r"(?:天气|气温|温度|下雨|下雪|冷不冷|热不热|多少度|几度|"
        r"怎么样|如何|咋样).*$",
        "",
        raw,
    )
    return clean_location_answer(raw)


def strip_time_prefix(name: str) -> str:
    return _TIME_PREFIX_RE.sub("", (name or "").strip()).strip()


# ---------------------------------------------------------------------------
# 意图识别
# ---------------------------------------------------------------------------


_STRONG_RE = re.compile(
    r"天气|气温|温度|几度|多少度|气象|预报|空气质量|雾霾|霾|紫外线|风力|风大|"
    r"下雨|下雪|下不下雨|会不会下雨|会不会下雪|带伞|雨伞|穿什么|穿多少|"
    r"外套|秋裤|羽绒服|短袖|冷不冷|热不热|降温|升温|寒潮|台风|冰雹|霜冻|"
    r"雷阵雨|回南天|梅雨|湿度|干燥|晒不晒|weather|temperature|forecast|umbrella",
    re.IGNORECASE,
)

_TIME_SCOPE_RE = {
    "day_after": re.compile(r"后天"),
    "tomorrow": re.compile(r"明天|明早|明晚|明儿|明日"),
    "week": re.compile(r"这周|本周|周末|下周|未来几天|这几天|接下来几天"),
    "now": re.compile(r"现在|当前|此刻|实时|马上|出门|外面"),
}

_STATEMENT_RE = re.compile(
    r"^(今天|刚才|外面)?(下雨|下雪|出太阳|起风|好冷|好热)了[。.!！~～]*$"
)
_PRAISE_RE = re.compile(r"天气(真|很|挺|好|不错|舒服|棒)")
_QUESTION_RE = re.compile(r"[?？]|吗|呢|怎么样|如何|咋样|会不会|要不要|多少")


def detect_weather_intent(text: str) -> bool:
    raw = (text or "").strip()
    if not raw:
        return False
    if raw.startswith(("/", "／")):
        return False
    if _STATEMENT_RE.match(raw):
        return False
    if _PRAISE_RE.search(raw) and not _QUESTION_RE.search(raw):
        return False
    if _STRONG_RE.search(raw):
        return True
    # 「冷吗」「热吗」这类短问句
    if re.search(r"(冷|热)(吗|不|么)", raw):
        return True
    return False


def detect_time_scope(text: str) -> str:
    raw = text or ""
    for scope in ("day_after", "tomorrow", "week", "now"):
        if _TIME_SCOPE_RE[scope].search(raw):
            return scope
    return "today"


def detect_aspects(text: str) -> set[str]:
    raw = text or ""
    aspects: set[str] = set()
    if re.search(r"下雨|雨|带伞|雨伞|淋|降水", raw):
        aspects.add("rain")
    if re.search(r"下雪|雪|冰|霜", raw):
        aspects.add("snow")
    if re.search(r"冷|热|温度|气温|几度|多少度|穿|保暖|外套", raw):
        aspects.add("temp")
    if re.search(r"空气|雾霾|霾|pm2\.?5|aqi", raw, re.IGNORECASE):
        aspects.add("air")
    if re.search(r"紫外线|晒|太阳|防晒", raw):
        aspects.add("uv")
    if re.search(r"风|台风|大风", raw):
        aspects.add("wind")
    return aspects


_TIME_PREFIX_RE = re.compile(
    r"^(今天|明天|后天|现在|最近|这两天|这几天|这周|本周|下周|未来|周末|早上|中午|晚上|白天|夜里)+"
)

_CITY_PATTERNS = [
    re.compile(r"^天气\s+([\u4e00-\u9fa5A-Za-z·]{2,10})"),
    re.compile(
        r"(?:查|查查|看看|看下|问问|问一下|想知道)?\s*"
        r"([\u4e00-\u9fa5]{2,8}?)(?:市|区|县)?(?:的)?"
        r"(?:今天|明天|后天|现在|最近|这周|下周|未来几天|周末)?"
        r"(?:会|要|会不会|可能)?"
        r"(?:天气|气温|温度|下雨|下雪|冷不冷|热不热|多少度|几度)"
    ),
    re.compile(r"在([\u4e00-\u9fa5]{2,8}?)(?:市|区|县)?"),
    re.compile(r"([\u4e00-\u9fa5]{2,8}?)(?:今天|明天)?(?:几点|什么时候)(?:开始)?(?:下|停)"),
    re.compile(r"([\u4e00-\u9fa5]{2,8}?)(?:雨|雪)(?:什么时候|几点)"),
]


def extract_city_candidates(text: str) -> list[str]:
    raw = (text or "").strip()
    if not raw:
        return []
    candidates: list[str] = []
    for pattern in _CITY_PATTERNS:
        for match in pattern.finditer(raw):
            name = match.group(1).strip()
            name = _TIME_PREFIX_RE.sub("", name).strip()
            name = re.sub(r"^(请问|帮我|给我|一下|的)+", "", name).strip()
            name = clean_location_answer(name)
            if not name or name in LOCATION_STOPWORDS:
                continue
            if len(name) < 2 or len(name) > 8:
                continue
            if re.search(r"天气|气温|温度|下雨|下雪|几度|多少度", name):
                continue
            if re.search(r"几点|什么时候|开始|停$", name):
                continue
            if name not in candidates:
                candidates.append(name)
    return candidates


_CANCEL_RE = re.compile(r"^(算了|不查了|不用了|取消|先不查|没事了|不用查了)[。.!！~～]*$")


def is_cancel(text: str) -> bool:
    return bool(_CANCEL_RE.match((text or "").strip()))


# ---------------------------------------------------------------------------
# 绑定存储
# ---------------------------------------------------------------------------


class BindingStore:
    """用户 -> 地区 的 JSON 持久化。"""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = str(path)
        self._data: dict[str, dict[str, Any]] | None = None
        self._lock = asyncio.Lock()

    async def _ensure(self) -> None:
        if self._data is not None:
            return
        async with self._lock:
            if self._data is not None:
                return

            def _read() -> dict[str, Any]:
                if not os.path.exists(self.path):
                    return {}
                try:
                    with open(self.path, encoding="utf-8") as fh:
                        payload = json.load(fh)
                except (OSError, json.JSONDecodeError):
                    return {}
                return payload if isinstance(payload, dict) else {}

            self._data = await asyncio.to_thread(_read)

    async def _save(self) -> None:
        assert self._data is not None
        payload = dict(self._data)

        def _write() -> None:
            directory = os.path.dirname(self.path) or "."
            os.makedirs(directory, exist_ok=True)
            # 不用 mkstemp：部分 Windows 环境下新建的临时目录权限异常
            tmp_path = self.path + ".tmp"
            try:
                with open(tmp_path, "w", encoding="utf-8") as fh:
                    json.dump(payload, fh, ensure_ascii=False, indent=2)
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp_path, self.path)
            finally:
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass

        await asyncio.to_thread(_write)

    async def get(self, user_key: str) -> dict[str, Any] | None:
        await self._ensure()
        assert self._data is not None
        record = self._data.get(str(user_key))
        return dict(record) if isinstance(record, dict) else None

    async def set(self, user_key: str, record: dict[str, Any]) -> None:
        await self._ensure()
        assert self._data is not None
        async with self._lock:
            self._data[str(user_key)] = dict(record)
            await self._save()

    async def delete(self, user_key: str) -> bool:
        await self._ensure()
        assert self._data is not None
        async with self._lock:
            existed = self._data.pop(str(user_key), None) is not None
            if existed:
                await self._save()
            return existed

    async def clear(self) -> int:
        await self._ensure()
        assert self._data is not None
        async with self._lock:
            count = len(self._data)
            self._data.clear()
            await self._save()
            return count

    async def all(self) -> dict[str, dict[str, Any]]:
        await self._ensure()
        assert self._data is not None
        return {key: dict(value) for key, value in self._data.items()}


# ---------------------------------------------------------------------------
# 多源服务
# ---------------------------------------------------------------------------


@dataclass
class ProviderHealth:
    fails: int = 0
    last_fail: float = 0.0
    last_ok: float = 0.0
    last_error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "fails": self.fails,
            "last_fail": self.last_fail,
            "last_ok": self.last_ok,
            "last_error": self.last_error,
        }


class WeatherService:
    """按配置顺序调用天气源，失败自动切换。"""

    def __init__(
        self,
        config_getter: Callable[[str, Any], Any],
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._cfg = config_getter
        self._clock = clock
        self.providers = {
            "cnweather": CNWeatherProvider(),
            "open_meteo": OpenMeteoProvider(),
            "wttr": WttrProvider(),
        }
        self.health: dict[str, ProviderHealth] = {
            key: ProviderHealth() for key in self.providers
        }
        self._cache: dict[str, tuple[float, WeatherData]] = {}
        self._geo_cache: dict[str, tuple[float, Location | None]] = {}
        self.stats = {"queries": 0, "cache_hits": 0, "failovers": 0, "errors": 0}
        self._lock = asyncio.Lock()

    # -- 配置 ---------------------------------------------------------
    def _order(self) -> list[str]:
        raw = self._cfg("sources.order", ["cnweather", "open_meteo", "wttr"])
        if isinstance(raw, str):
            raw = [item.strip() for item in raw.split(",") if item.strip()]
        if not isinstance(raw, (list, tuple)):
            raw = []
        order = [item for item in raw if item in self.providers]
        for key in self.providers:
            if key not in order:
                order.append(key)
        return order

    def _timeout(self) -> float:
        try:
            return max(2.0, float(self._cfg("sources.timeout", 6)))
        except (TypeError, ValueError):
            return 6.0

    def _retries(self) -> int:
        try:
            return max(0, min(3, int(self._cfg("sources.retries", 1))))
        except (TypeError, ValueError):
            return 1

    def _cooldown(self) -> float:
        try:
            return max(0.0, float(self._cfg("sources.cooldown", 300)))
        except (TypeError, ValueError):
            return 300.0

    def _cache_ttl(self) -> float:
        try:
            return max(0.0, float(self._cfg("sources.cache_ttl", 600)))
        except (TypeError, ValueError):
            return 600.0

    def _days(self) -> int:
        try:
            return max(1, min(7, int(self._cfg("reply.forecast_days", 3))))
        except (TypeError, ValueError):
            return 3

    def _show_air(self) -> bool:
        return bool(self._cfg("features.show_air_quality", True))

    # -- 地点 ---------------------------------------------------------
    async def resolve(self, name: str, *, force: bool = False) -> Location | None:
        cleaned = clean_location_answer(name)
        if not cleaned:
            return None
        cache_key = cleaned.lower()
        now = self._clock()
        if not force:
            hit = self._geo_cache.get(cache_key)
            if hit and hit[0] > now:
                return hit[1]

        timeout = self._timeout()
        retries = self._retries()
        province_hint, _ = _split_province(_normalize_location_name(cleaned))

        location: Location | None = None
        # 「杭州市」「海淀区」这类带后缀的，先按原样试，不理想再试去掉后缀
        variants = self._name_variants(cleaned)
        for index, variant in enumerate(variants):
            cn_results, geo = await self._lookup_name(variant, timeout, retries)
            best = _pick_cn_result(cn_results, variant, province_hint)
            if best is not None:
                lat: float | None = None
                lon: float | None = None
                # 只有省份对得上时才采用 Open-Meteo 的经纬度，避免「朝阳区」
                # 拿到辽宁的坐标却报北京的天气
                if geo is not None and _provinces_match(
                    best.get("province"), geo.get("province")
                ):
                    lat = _to_float(geo.get("lat"))
                    lon = _to_float(geo.get("lon"))
                location = Location(
                    name=str(best.get("name") or variant),
                    province=str(best.get("province") or ""),
                    lat=lat,
                    lon=lon,
                    cn_code=str(best.get("code") or ""),
                )
                break
            if index == len(variants) - 1 and geo is not None and geo.get("lat") is not None:
                # 最后一个候选仍没有中国城市代码，就用 Open-Meteo（国外城市）
                location = Location(
                    name=str(geo.get("name") or variant),
                    province=str(geo.get("province") or ""),
                    lat=_to_float(geo.get("lat")),
                    lon=_to_float(geo.get("lon")),
                )
                break

        if location is not None and (location.cn_code or location.lat is not None):
            ttl = 7 * 86400 if not force else 0
            if len(self._geo_cache) > 512:
                self._geo_cache.clear()
            self._geo_cache[cache_key] = (now + ttl, location)
        else:
            # 解析失败只缓存 5 分钟，避免一次网络抖动让用户一小时都查不了
            self._geo_cache[cache_key] = (now + 300, None)
            location = None
        return location

    @staticmethod
    def _name_variants(name: str) -> list[str]:
        return chinese_location_variants(name)

    async def _lookup_name(
        self, name: str, timeout: float, retries: int
    ) -> tuple[list[dict[str, str]], dict[str, Any] | None]:
        cn_provider = self.providers["cnweather"]
        om_provider = self.providers["open_meteo"]
        cn_task = asyncio.create_task(
            cn_provider.search(name, timeout=timeout, retries=retries)
        )
        geo_task = asyncio.create_task(
            om_provider.geocode(name, timeout=timeout, retries=retries)
        )
        try:
            cn_value, geo_value = await asyncio.gather(
                cn_task, geo_task, return_exceptions=True
            )
        finally:
            for task in (cn_task, geo_task):
                if not task.done():
                    task.cancel()
        cn_results = cn_value if isinstance(cn_value, list) else []
        geo = geo_value if isinstance(geo_value, dict) else None
        return cn_results, geo

    # -- 天气 ---------------------------------------------------------
    async def get(
        self,
        location: Location,
        *,
        force_refresh: bool = False,
    ) -> WeatherData:
        days = self._days()
        cache_key = f"{location.key}|{days}|{self._show_air()}"
        now = self._clock()
        if not force_refresh:
            cached = self._cache.get(cache_key)
            if cached and cached[0] > now:
                self.stats["cache_hits"] += 1
                return cached[1]

        self.stats["queries"] += 1
        timeout = self._timeout()
        retries = self._retries()
        cooldown = self._cooldown()
        errors: list[str] = []
        tried = 0

        for provider_id in self._order():
            provider = self.providers.get(provider_id)
            if provider is None:
                continue
            health = self.health[provider_id]
            if cooldown and health.last_fail and now - health.last_fail < cooldown:
                errors.append(f"{provider.label}: 冷却中")
                continue
            if not self._can_use(provider_id, location):
                errors.append(f"{provider.label}: 缺少定位信息")
                continue
            tried += 1
            try:
                data = await self._call(provider, location, timeout, retries)
                health.fails = 0
                health.last_ok = now
                health.last_error = ""
                ttl = self._cache_ttl()
                if ttl > 0:
                    if len(self._cache) > 512:
                        self._cache.clear()
                    self._cache[cache_key] = (now + ttl, data)
                if tried > 1:
                    self.stats["failovers"] += 1
                return data
            except Exception as exc:  # noqa: BLE001
                health.fails += 1
                health.last_fail = now
                health.last_error = str(exc)[:200]
                errors.append(f"{provider.label}: {exc}")
                self.stats["errors"] += 1

        raise WeatherError("天气源都没能返回数据（" + "；".join(errors) + "）")

    def _can_use(self, provider_id: str, location: Location) -> bool:
        if provider_id == "cnweather":
            return bool(location.cn_code or location.name)
        if provider_id == "open_meteo":
            return location.lat is not None and location.lon is not None
        if provider_id == "wttr":
            return bool(location.name) or (
                location.lat is not None and location.lon is not None
            )
        return True

    async def _call(
        self,
        provider: Any,
        location: Location,
        timeout: float,
        retries: int,
    ) -> WeatherData:
        kwargs: dict[str, Any] = {
            "days": self._days(),
            "timeout": timeout,
            "retries": retries,
        }
        if provider.id == "open_meteo":
            kwargs["include_air"] = self._show_air()
        data = await asyncio.wait_for(
            provider.fetch(location, **kwargs), timeout=timeout + 1.5
        )
        if not isinstance(data, WeatherData):
            raise ProviderError("数据源返回了意外的结果")
        return data

    async def hourly(
        self, location: Location, *, hours: int = 12
    ) -> tuple[str, list[HourlyPoint]]:
        """小时级降水：优先 Open-Meteo，失败用 wttr。返回 (来源, 数据点)。"""
        timeout = self._timeout()
        retries = self._retries()
        target = location
        if target.lat is None or target.lon is None:
            try:
                geo = await self.providers["open_meteo"].geocode(
                    target.name, timeout=timeout, retries=retries
                )
            except Exception:  # noqa: BLE001
                geo = None
            if geo and geo.get("lat") is not None:
                target = Location(
                    name=str(geo.get("name") or target.name),
                    province=str(geo.get("province") or target.province),
                    lat=_to_float(geo.get("lat")),
                    lon=_to_float(geo.get("lon")),
                    cn_code=target.cn_code,
                )
        errors: list[str] = []
        if target.lat is not None and target.lon is not None:
            try:
                points = await self.providers["open_meteo"].fetch_hourly(
                    target, timeout=timeout, retries=retries, hours=hours
                )
                if points:
                    return "Open-Meteo", points
            except Exception as exc:  # noqa: BLE001
                errors.append(f"Open-Meteo: {exc}")
        try:
            points = await self.providers["wttr"].fetch_hourly(
                target, timeout=timeout, retries=retries, hours=hours
            )
            if points:
                return "wttr.in", points
        except Exception as exc:  # noqa: BLE001
            errors.append(f"wttr.in: {exc}")
        raise WeatherError("小时级预报暂时拿不到（" + "；".join(errors) + "）")

    # -- 体检 ---------------------------------------------------------
    def source_state(self) -> list[dict[str, Any]]:
        now = self._clock()
        cooldown = self._cooldown()
        result = []
        for idx, provider_id in enumerate(self._order()):
            provider = self.providers[provider_id]
            health = self.health[provider_id]
            if cooldown and health.last_fail and now - health.last_fail < cooldown:
                state = "cooldown"
            elif health.last_fail and not health.last_ok:
                state = "down"
            elif health.last_ok:
                state = "ok"
            else:
                state = "idle"
            result.append(
                {
                    "id": provider_id,
                    "label": provider.label,
                    "order": idx + 1,
                    "state": state,
                    "cooldown_left": max(
                        0, int(cooldown - (now - health.last_fail))
                    ),
                    **health.to_dict(),
                }
            )
        return result

    def cache_size(self) -> int:
        return len(self._cache)

    def clear_cache(self) -> None:
        self._cache.clear()
        self._geo_cache.clear()


_CN_CITY_CODE_RE = re.compile(r"^\d{9}$")
_PROVINCE_SUFFIX_RE = re.compile(
    r"(壮族自治区|回族自治区|维吾尔自治区|特别行政区|自治区|省|市)$"
)


def _normalize_province(value: Any) -> str:
    text = str(value or "").strip()
    return _PROVINCE_SUFFIX_RE.sub("", text)


def _provinces_match(left: Any, right: Any) -> bool:
    a = _normalize_province(left)
    b = _normalize_province(right)
    if not a or not b:
        return False
    return a == b or a.startswith(b) or b.startswith(a)


def _pick_cn_result(
    results: list[dict[str, str]],
    query: str,
    preferred_province: str = "",
) -> dict[str, str] | None:
    """只认 9 位城市/区县代码，过滤掉「景点」「街道」等结果。"""
    valid = [
        item
        for item in results
        if _CN_CITY_CODE_RE.match(str(item.get("code") or ""))
    ]
    if not valid:
        return None
    if preferred_province:
        matched = [
            item
            for item in valid
            if _provinces_match(item.get("province"), preferred_province)
        ]
        if matched:
            valid = matched
    bases = [query]
    for suffix in ("市", "区", "县", "省"):
        if query.endswith(suffix) and len(query) > len(suffix) + 1:
            bases.append(query[: -len(suffix)])
    for base in bases:
        for item in valid:
            if item.get("name") == base:
                return item
    for base in bases:
        for item in valid:
            if str(item.get("name") or "").startswith(base):
                return item
    return valid[0]


# ---------------------------------------------------------------------------
# 回复模板（人格润色失败时的兜底）
# ---------------------------------------------------------------------------


ASK_TEMPLATES = [
    "你在哪个城市呀\n说个城市名给我，以后我就记着啦",
    "先告诉我你在哪儿嘛\n不然我查不到天气",
    "报个城市名呗\n就这一次，之后不用再说啦",
]


def ask_location_template() -> str:
    return random.choice(ASK_TEMPLATES)


def advice_lines(
    data: WeatherData,
    aspects: set[str] | None = None,
    scope: str = "today",
    *,
    show_air: bool = True,
) -> list[str]:
    aspects = aspects or set()
    advices: list[str] = []

    if scope in ("tomorrow", "day_after"):
        index = 1 if scope == "tomorrow" else 2
        target = data.daily[index] if len(data.daily) > index else data.today
    else:
        target = data.today

    day_text = (target.text if target else "") or ""
    night_text = (target.text_night if target else "") or ""
    texts = " ".join([data.current_text or "", day_text, night_text])
    precip = target.precip_prob if target else None
    rain_expected = _is_rainy(texts) or (precip is not None and precip >= 50)
    snow_expected = "雪" in texts or "冰" in texts

    if snow_expected:
        advices.append("有雪，路上小心")
    elif rain_expected:
        advices.append("记得带伞")
    elif "rain" in aspects:
        advices.append("应该不下雨，不带伞也行")

    high = target.temp_max if target else data.today_max
    low = target.temp_min if target else data.today_min
    if high is not None and high >= 30:
        advices.append("挺热的，注意防晒")
    if low is not None and low <= 3:
        advices.append("有点冷，多穿点")
    if show_air and data.aqi is not None and data.aqi >= 150:
        advices.append("空气不太好，戴口罩")
    if target is not None and target.wind_scale and _scale_value(target.wind_scale) >= 5:
        advices.append("风有点大")
    if data.humidity is not None and data.humidity >= 90 and not rain_expected:
        advices.append("有点潮")
    # 去重，最多两条
    unique: list[str] = []
    for item in advices:
        if item not in unique:
            unique.append(item)
    return unique[:2]


def _scale_value(scale: str) -> int:
    match = re.search(r"(\d+)", scale or "")
    return int(match.group(1)) if match else 0


def render_template(
    data: WeatherData,
    scope: str = "today",
    aspects: set[str] | None = None,
    *,
    show_air: bool = True,
) -> str:
    lines: list[str] = []
    name = data.location.name
    if scope in ("tomorrow", "day_after"):
        index = 1 if scope == "tomorrow" else 2
        target = data.daily[index] if len(data.daily) > index else data.today
        if target is not None:
            label = target.label or ("明天" if index == 1 else "后天")
            target_text = target.text or ""
            if target.text_night and target.text_night != target.text:
                target_text = f"{target_text}转{target.text_night}" if target_text else target.text_night
            line = f"{name}{label}"
            if target_text:
                line += f"，{target_text}"
            if target.temp_min is not None and target.temp_max is not None:
                line += f"，{format_temp(target.temp_min)}~{format_temp(target.temp_max)}℃"
            lines.append(line)
            if target.precip_prob is not None and target.precip_prob >= 20:
                lines.append(f"降水概率{target.precip_prob}%")
    else:
        current = f"{name}现在"
        if data.current_text:
            current += data.current_text
        if data.current_temp is not None:
            current += f" {format_temp(data.current_temp)}℃"
        lines.append(current)
        if data.today_min is not None and data.today_max is not None:
            today_line = f"今天{format_temp(data.today_min)}~{format_temp(data.today_max)}℃"
            if data.today and data.today.text:
                today_line = f"{data.today.text}，{today_line}"
            lines.append(today_line)
        if (
            data.today
            and data.today.precip_prob is not None
            and data.today.precip_prob >= 30
        ):
            lines.append(f"降水概率{data.today.precip_prob}%")
        if show_air and data.aqi_text:
            lines.append(f"空气{data.aqi_text}")

    advices = advice_lines(data, aspects, scope, show_air=show_air)
    if advices:
        lines.append("，".join(advices))
    return "\n".join(line for line in lines if line.strip())


# ---------------------------------------------------------------------------
# 给 LLM 的提示词
# ---------------------------------------------------------------------------


def build_ask_prompt(nickname: str = "", max_chars: int = 120) -> str:
    who = f"（对方叫「{nickname}」）" if nickname else ""
    return (
        "【场景】对方刚刚问天气，但你还不知道他在哪个城市。\n"
        f"{who}\n"
        "【要求】\n"
        "- 用你平时说话的口吻，随口问一句他在哪个城市\n"
        "- 可以顺带说一句「记住以后就不问了」\n"
        "- 最多两行，每行不超过30个字，不用句号\n"
        "- 不要像表单，不要说「绑定」「设置」「系统」「请输入」\n"
        "- 只输出你要对他说的那句话\n"
        f"控制在{max_chars}字以内。"
    )


def build_bind_prompt(city: str, nickname: str = "", max_chars: int = 120) -> str:
    who = f"（对方叫「{nickname}」）" if nickname else ""
    return (
        f"【场景】对方让你以后按「{city}」查天气，你已经记下了。\n"
        f"{who}\n"
        "【要求】\n"
        "- 用你平时说话的口吻应一声，表示记住了\n"
        "- 1到2行，每行不超过30个字，不用句号\n"
        "- 不要像客服，不要提「绑定」「设置」「系统」\n"
        "- 只输出你要对他说的那句话\n"
        f"控制在{max_chars}字以内。"
    )


def build_reply_prompt(
    data: WeatherData,
    user_text: str,
    *,
    scope: str = "today",
    aspects: set[str] | None = None,
    nickname: str = "",
    max_chars: int = 200,
    first_time: bool = False,
    extra: str = "",
    show_alerts: bool = True,
    show_air: bool = True,
    show_indices: bool = True,
    mood: str = "",
    snapshot_note: str = "",
) -> str:
    aspect_labels = {
        "rain": "会不会下雨、要不要带伞",
        "snow": "会不会下雪、路上滑不滑",
        "temp": "冷热、穿什么",
        "air": "空气质量、要不要戴口罩",
        "uv": "晒不晒、要不要防晒",
        "wind": "风大不大",
    }
    scope_labels = {
        "today": "今天/现在",
        "now": "现在/马上出门",
        "tomorrow": "明天",
        "day_after": "后天",
        "week": "未来几天",
    }
    focus = "、".join(aspect_labels[a] for a in sorted(aspects or set()) if a in aspect_labels)
    if not focus:
        focus = "现在怎么样、今天大概什么温度"

    lines = [
        "【天气数据】",
        data.to_prompt_text(
            max_days=4,
            show_alerts=show_alerts,
            show_air=show_air,
            aspects=aspects,
            show_indices=show_indices,
        ),
        "",
        f"【对方刚刚说】{(user_text or '')[:200]}",
        f"【他关心】{focus}；时间范围：{scope_labels.get(scope, '今天')}",
    ]
    if mood:
        lines.append(f"【你现在的状态】{mood}")
    if snapshot_note:
        lines.append(f"【和上次对比】{snapshot_note}")
    lines.extend(
        [
            "",
            "【要求】",
            "- 用你平时说话的口吻，像在 QQ 上随手回消息",
            "- 1到3行，每行不超过30个字，行与行之间换行，不用句号",
            "- 必须提到城市，以及关键天气（温度、下不下雨/雪）",
            "- 不要罗列字段，不要用 markdown、分点、标题、括号补充",
            "- 不要说「根据数据」「天气 API」「数据源」",
            "- 不要解释你在查天气，直接说结果",
        ]
    )
    if first_time:
        lines.append(f"- 这是第一次给他记下城市，开头顺口提一句你记住了{data.location.name}")
    if extra:
        lines.append(f"- {extra}")
    lines.append(f"- 只输出你要对他说的话，控制在{max_chars}字以内")
    return "\n".join(lines)


def clean_llm_reply(text: str, max_chars: int = 200) -> str:
    """清理 LLM 输出：去引号、去前缀、去 markdown、限长。"""
    if not text:
        return ""
    content = text.strip()
    content = re.sub(r"^```[a-zA-Z]*\s*", "", content)
    content = re.sub(r"\s*```$", "", content)
    content = content.strip()
    # 去掉包裹引号
    if len(content) >= 2 and content[0] in "“\"'‘「" and content[-1] in "”\"'’」":
        content = content[1:-1].strip()
    # 去掉常见前缀
    content = re.sub(
        r"^(好的|好呀|嗯嗯|收到|回复|回答)[，,：:\s]+", "", content
    ).strip()
    # 去掉 markdown 标题/列表符号
    content = re.sub(r"^#{1,6}\s*", "", content, flags=re.MULTILINE)
    content = re.sub(r"^[-*]\s+", "", content, flags=re.MULTILINE)
    # 压缩空行
    content = re.sub(r"\n{2,}", "\n", content).strip()
    # 去掉整段括号旁白
    content = re.sub(r"^（[^）]*）$", "", content, flags=re.MULTILINE).strip()
    if len(content) > max_chars:
        cut = content[:max_chars]
        for mark in ("。", "！", "？", "\n", "，", " "):
            idx = cut.rfind(mark)
            if idx >= max_chars * 0.5:
                cut = cut[:idx]
                break
        content = cut.rstrip()
    return content



# ---------------------------------------------------------------------------
# 生活指数 / 心情 / 对比 / 节气
# ---------------------------------------------------------------------------


def pick_indices(
    data: WeatherData, aspects: set[str] | None = None, limit: int = 2
) -> list[tuple[str, str, str]]:
    """按当前关注点挑 1-2 条生活指数，返回 (名称, 提示, 说明)。"""
    if not data.indices:
        return []
    aspects = aspects or set()
    keys: list[str] = []
    if "rain" in aspects or "snow" in aspects:
        keys.extend(["lk", "ls", "gj"])
    if "uv" in aspects:
        keys.extend(["uv", "gl"])
    if "temp" in aspects:
        keys.extend(["ct", "co", "wc"])
    if "air" in aspects:
        keys.append("pl")
    keys.extend(["ct", "co", "uv"])
    seen: set[str] = set()
    result: list[tuple[str, str, str]] = []
    for key in keys:
        if key in seen:
            continue
        item = data.indices.get(key)
        if not item:
            continue
        name = str(item.get("name") or "").replace("指数", "").strip()
        hint = str(item.get("hint") or "").strip()
        if not name or not hint:
            continue
        seen.add(key)
        result.append((name, hint, str(item.get("desc") or "").strip()))
        if len(result) >= limit:
            break
    return result


def mood_hint(data: WeatherData, hour: int | None = None) -> str:
    """给润色提示词加一句“她现在的状态”，让语气跟着天气走。"""
    now_hour = time.localtime().tm_hour if hour is None else hour
    current = data.current_text or ""
    today_text = (data.today.text if data.today else "") or ""
    primary = current or today_text
    if now_hour >= 23 or now_hour < 6:
        return "这个点你已经有点困了，说话懒懒的"
    if "雪" in primary:
        return "外面下雪，你有点兴奋，又怕冷"
    if _is_rainy(primary):
        return "下雨天，你懒懒的，不太想动"
    if "晴" in primary:
        if 8 <= now_hour < 18:
            return "今天太阳很好，你心情不错"
        return "晚上天晴，你有点想出去走走"
    if "阴" in primary or "多云" in primary:
        return "天有点阴，你提不起劲"
    return ""


def compare_snapshot(data: WeatherData, snapshot: dict[str, Any] | None) -> str:
    """和上次查询对比，返回一句人能听懂的话。"""
    if not snapshot:
        return ""
    if snapshot.get("city") and snapshot["city"] != data.location.name:
        return ""
    previous = _to_float(snapshot.get("temp"))
    current = data.current_temp
    if previous is None or current is None:
        return ""
    diff = current - previous
    if abs(diff) < 3:
        return ""
    direction = "暖" if diff > 0 else "冷"
    return f"比上次{direction}了{abs(round(diff))}度"


_SOLAR_TERMS: list[tuple[int, float, str]] = [
    (1, 5.4055, "小寒"),
    (1, 20.12, "大寒"),
    (2, 3.87, "立春"),
    (2, 18.73, "雨水"),
    (3, 5.63, "惊蛰"),
    (3, 20.646, "春分"),
    (4, 4.81, "清明"),
    (4, 20.1, "谷雨"),
    (5, 5.52, "立夏"),
    (5, 21.04, "小满"),
    (6, 5.678, "芒种"),
    (6, 21.37, "夏至"),
    (7, 7.108, "小暑"),
    (7, 22.83, "大暑"),
    (8, 7.5, "立秋"),
    (8, 23.13, "处暑"),
    (9, 7.646, "白露"),
    (9, 23.042, "秋分"),
    (10, 8.318, "寒露"),
    (10, 23.438, "霜降"),
    (11, 7.438, "立冬"),
    (11, 22.36, "小雪"),
    (12, 7.18, "大雪"),
    (12, 21.94, "冬至"),
]


def solar_term_today(day: _dt.date | None = None) -> str:
    """二十四节气（21 世纪常用近似算法，个别年份可能差一天）。"""
    target = day or _dt.date.today()
    y = target.year % 100
    for month, constant, name in _SOLAR_TERMS:
        if month != target.month:
            continue
        expected = int(y * 0.2422 + constant) - int((y - 1) / 4)
        if target.day == expected:
            return name
    return ""


def pick_meme_kind(data: WeatherData) -> str:
    texts = " ".join(
        [
            data.current_text or "",
            (data.today.text if data.today else "") or "",
        ]
    )
    if "雪" in texts:
        return "snow"
    if _is_rainy(texts):
        return "rain"
    if data.current_temp is not None and data.current_temp >= 32:
        return "hot"
    if data.current_temp is not None and data.current_temp <= 0:
        return "cold"
    if "晴" in texts:
        return "sunny"
    if "阴" in texts or "多云" in texts:
        return "cloudy"
    return ""


# ---------------------------------------------------------------------------
# 反向地理编码（QQ 位置卡片）
# ---------------------------------------------------------------------------


def _first_value(items: Any) -> str:
    if isinstance(items, list) and items:
        first = items[0]
        if isinstance(first, dict):
            return str(first.get("value") or "").strip()
    return ""


async def reverse_geocode(
    lat: float, lon: float, *, timeout: float = 8, retries: int = 1
) -> Location | None:
    """经纬度 -> 城市。优先 BigDataCloud（中文），失败用 wttr 最近地区。"""
    try:
        url = (
            "https://api.bigdatacloud.net/data/reverse-geocode-client"
            f"?latitude={lat}&longitude={lon}&localityLanguage=zh"
        )
        payload = await asyncio.to_thread(
            _get_json, url, timeout=timeout, retries=retries
        )
        if isinstance(payload, dict):
            name = str(
                payload.get("city")
                or payload.get("locality")
                or payload.get("principalSubdivision")
                or ""
            ).strip()
            if name:
                return Location(
                    name=name,
                    province=str(payload.get("principalSubdivision") or "").strip(),
                    lat=lat,
                    lon=lon,
                )
    except Exception:  # noqa: BLE001 - 换备用
        pass
    try:
        url = f"https://wttr.in/{lat:.4f},{lon:.4f}?format=j1&lang=zh"
        payload = await asyncio.to_thread(
            _get_json, url, timeout=timeout, retries=retries
        )
        nearest = (payload.get("nearest_area") or [{}])[0] if isinstance(payload, dict) else {}
        name = _first_value(nearest.get("areaName"))
        if name:
            return Location(
                name=name,
                province=_first_value(nearest.get("region")),
                lat=lat,
                lon=lon,
            )
    except Exception:  # noqa: BLE001
        pass
    return None


# ---------------------------------------------------------------------------
# 追问 / 出门 / 小时级降水 / 确认 / 别名
# ---------------------------------------------------------------------------


_FOLLOWUP_RE = re.compile(
    r"^(?:那|那么|然后)?\s*(今天|明天|后天|这周|本周|周末|下周)?\s*"
    r"(?:呢|怎么样|如何|咋样|天气(?:怎么样|如何|咋样|呢)?|会下吗|会下(?:雨|雪)吗|"
    r"下雨吗|下雪吗|冷不冷|热不热|几度|多少度|温度呢?|气温呢?)[?？]?$"
)

_DEPARTURE_RE = re.compile(
    r"^(?:我)?(?:要|准备|马上|现在)?(?:去)?"
    r"(?:出门|出发|走了|上班|上学|接孩子|下楼|出去|通勤|回家)"
    r"(?:去了|了|啦|呀|咯)?[。.!！~～]*$"
)

_HOURLY_RE = re.compile(
    r"几点(?:开始)?(?:下|停)|什么时候(?:开始)?(?:下|停)|"
    r"(?:雨|雪)(?:什么时候|几点)(?:停|下)|会下多久|还要下多久|下到几点"
)

_YES_RE = re.compile(r"^(是|对|嗯|是的|对的|没错|可以|好|行|嗯嗯|是呀|对啊)[。.!！~～]*$")
_NO_RE = re.compile(r"^(不是|不对|不|错了|不是的|不对)[。.!！~～]*$")

_ALIAS_RE = re.compile(
    r"(?:记住|以后|以后都)?\s*"
    r"(老家|老家那边|公司|公司那边|学校|单位|家|家里|爸妈家|娘家|婆家|新家)"
    r"\s*(?:是|在|=|改成|改到)\s*([\u4e00-\u9fa5A-Za-z·]{2,10})"
)

_CUSTOM_ALIAS_RE = re.compile(
    r"记住\s*([\u4e00-\u9fa5]{1,6})\s*(?:是|在|=)\s*([\u4e00-\u9fa5A-Za-z·]{2,10})"
)


def detect_followup(text: str) -> dict[str, Any] | None:
    """识别「那明天呢」「会下吗」这类追问。"""
    raw = (text or "").strip()
    if not raw or len(raw) > 16:
        return None
    if not _FOLLOWUP_RE.match(raw):
        return None
    aspects = detect_aspects(raw)
    if re.search(r"会下|下吗|下雨|下雪", raw):
        aspects.add("rain")
    if re.search(r"冷|热|几度|多少度|温度|气温", raw):
        aspects.add("temp")
    return {
        "scope": detect_time_scope(raw),
        "aspects": aspects,
    }


def is_departure(text: str) -> bool:
    return bool(_DEPARTURE_RE.match((text or "").strip()))


def detect_hourly_intent(text: str) -> bool:
    return bool(_HOURLY_RE.search(text or ""))


def is_confirm_yes(text: str) -> bool:
    return bool(_YES_RE.match((text or "").strip()))


def is_confirm_no(text: str) -> bool:
    return bool(_NO_RE.match((text or "").strip()))


def extract_alias_binding(text: str) -> tuple[str, str] | None:
    """「记住老家是临沂」「老家在临沂」 -> (别名, 城市)。"""
    raw = (text or "").strip()
    match = _ALIAS_RE.search(raw)
    if match:
        alias = match.group(1).strip().replace("那边", "").strip()
        return alias, match.group(2).strip()
    match = _CUSTOM_ALIAS_RE.search(raw)
    if match:
        alias = match.group(1).strip()
        if (
            len(alias) >= 1
            and alias not in LOCATION_STOPWORDS
            and not re.search(r"[在是我你他她]", alias)
            and not re.search(r"天气|气温|城市|地方|什么|怎么", alias)
        ):
            return alias, match.group(2).strip()
    return None


# ---------------------------------------------------------------------------
# 小时级降水
# ---------------------------------------------------------------------------


def summarize_hourly(points: list[HourlyPoint]) -> str:
    if not points:
        return ""
    first = points[0]
    wet = [
        p
        for p in points
        if (p.precip_prob is not None and p.precip_prob >= 50)
        or (p.precip_mm is not None and p.precip_mm > 0.1)
        or _is_rainy(p.text)
    ]
    if not wet:
        return "接下来几小时应该没雨"
    if _is_rainy(first.text) or (
        first.precip_prob is not None and first.precip_prob >= 50
    ):
        dry = next(
            (
                p
                for p in points
                if not _is_rainy(p.text)
                and (p.precip_prob is None or p.precip_prob < 40)
            ),
            None,
        )
        if dry is not None:
            return f"现在在下，{dry.time} 左右会停"
        return "这雨一时半会儿停不了"
    start = wet[0]
    end = None
    started = False
    for p in points:
        is_wet = (
            (p.precip_prob is not None and p.precip_prob >= 50)
            or (p.precip_mm is not None and p.precip_mm > 0.1)
            or _is_rainy(p.text)
        )
        if is_wet:
            started = True
        elif started and end is None:
            end = p
    if end is not None:
        return f"{start.time} 左右开始下，{end.time} 左右停"
    return f"{start.time} 左右开始下"


# ---------------------------------------------------------------------------
# 主动提醒 / 失败 / 出门 / 确认 / 猜城市 的提示词
# ---------------------------------------------------------------------------


def reminder_reason(
    data: WeatherData,
    snapshot: dict[str, Any] | None = None,
    temp_drop: int = 6,
) -> str:
    if data.alerts:
        return "有气象预警"
    texts = " ".join(
        [
            data.current_text or "",
            (data.today.text if data.today else "") or "",
            (data.today.text_night if data.today else "") or "",
        ]
    )
    if "暴雨" in texts or "大雨" in texts:
        return "有大雨"
    if "雪" in texts:
        return "有雪"
    prev_high = _to_float((snapshot or {}).get("high"))
    prev_low = _to_float((snapshot or {}).get("low"))
    drops: list[float] = []
    if prev_high is not None and data.today_max is not None:
        drops.append(prev_high - data.today_max)
    if prev_low is not None and data.today_min is not None:
        drops.append(prev_low - data.today_min)
    if drops and max(drops) >= max(1, int(temp_drop)):
        return f"降温{round(max(drops))}度"
    if data.today_max is not None and data.today_max >= 35:
        return "高温"
    if data.today_min is not None and data.today_min <= -5:
        return "严寒"
    if data.aqi is not None and data.aqi >= 200:
        return "空气很差"
    return ""


def reminder_template(data: WeatherData, reason: str) -> str:
    name = data.location.name
    if "雨" in reason:
        tail = "记得带伞"
    elif "雪" in reason:
        tail = "路上小心，穿暖一点"
    elif "降温" in reason:
        tail = "多穿点，别冻着"
    elif "高温" in reason:
        tail = "少在太阳底下久待"
    elif "空气" in reason:
        tail = "出门戴个口罩"
    elif "预警" in reason:
        tail = "注意安全"
    else:
        tail = "注意天气变化"
    bits = [f"{name}今天{data.today.text}" if data.today and data.today.text else name]
    if data.today_min is not None and data.today_max is not None:
        bits.append(f"{format_temp(data.today_min)}~{format_temp(data.today_max)}℃")
    return f"{'，'.join(bits)}\n{tail}"


def build_departure_prompt(
    data: WeatherData,
    user_text: str,
    *,
    nickname: str = "",
    max_chars: int = 120,
    extra: str = "",
) -> str:
    return (
        "【场景】对方正准备出门，没问你天气，你想顺口提醒他一句。\n"
        f"【天气数据】\n{data.to_prompt_text(max_days=1, aspects=set(), show_indices=False)}\n\n"
        f"【他刚刚说】{user_text[:100]}\n"
        "【要求】\n"
        "- 像随口关心一句，不要像天气预报\n"
        "- 1到2行，每行不超过30字，不用句号\n"
        "- 必须提到现在的温度或天气，以及一个具体提醒（带伞/外套/防晒）\n"
        "- 只输出你要对他说的话\n"
        f"- 控制在{max_chars}字以内"
        + (f"\n- {extra}" if extra else "")
    )


def build_hourly_prompt(
    data: WeatherData,
    points: list[HourlyPoint],
    user_text: str,
    *,
    max_chars: int = 120,
    extra: str = "",
) -> str:
    lines = [f"【城市】{data.location.name}", "【未来几小时】"]
    for point in points[:8]:
        bit = f"{point.time} {point.text}".strip()
        if point.temp is not None:
            bit += f" {format_temp(point.temp)}℃"
        if point.precip_prob is not None:
            bit += f" 降水{point.precip_prob}%"
        lines.append(bit)
    lines.extend(
        [
            f"【他问】{user_text[:100]}",
            "【要求】",
            "- 直接回答几点下雨/几点停，别报一长串数据",
            "- 1到2行，每行不超过30字，不用句号\n"
            "- 只输出你要对他说的话",
            f"- 控制在{max_chars}字以内",
        ]
    )
    if extra:
        lines.append(f"- {extra}")
    return "\n".join(lines)


def build_city_guess_prompt(raw: str) -> str:
    return (
        "用户输入了一个可能是城市名的词，但天气接口没查到。\n"
        f"用户输入：{raw[:40]}\n"
        "请猜他想说的是哪个中国城市（考虑同音字、错别字、拼音首字母）。\n"
        "只输出一个城市名（2到6个汉字），不要解释；猜不出就输出「无」。"
    )


def build_confirm_city_prompt(city: str, nickname: str = "") -> str:
    who = f"（对方叫「{nickname}」）" if nickname else ""
    return (
        f"【场景】对方可能把「{city}」打错了，你想确认一下。\n"
        f"{who}\n"
        "【要求】\n"
        "- 用你平时的口吻问一句「你是说{city}吗」\n"
        "- 只问一次，1行，不超过30字，不用句号\n"
        "- 不要解释，不要提接口\n"
        "- 只输出你要说的话"
    )


def build_reminder_prompt(
    data: WeatherData,
    reason: str,
    *,
    nickname: str = "",
    max_chars: int = 120,
    extra: str = "",
) -> str:
    who = f"（对方叫「{nickname}」）" if nickname else ""
    return (
        "【场景】对方今天还没问天气，你想主动提醒他一句。\n"
        f"{who}\n"
        f"【触发原因】{reason}\n"
        f"【天气数据】\n{data.to_prompt_text(max_days=2, show_indices=True)}\n\n"
        "【要求】\n"
        "- 像随口提醒，不要像通知，不要每天固定话术\n"
        "- 1到2行，每行不超过30字，不用句号\n"
        "- 必须包含城市和一个具体建议\n"
        "- 只输出你要对他说的话\n"
        f"- 控制在{max_chars}字以内"
        + (f"\n- {extra}" if extra else "")
    )


def build_failure_prompt(error: str, nickname: str = "") -> str:
    who = f"（对方叫「{nickname}」）" if nickname else ""
    return (
        "【场景】对方问天气，但所有天气接口都没连上。\n"
        f"{who}\n"
        f"【技术原因】{str(error)[:120]}\n"
        "【要求】\n"
        "- 用你平时的口吻道个歉，像网络抽风那种，不要报错代码\n"
        "- 1行，不超过30字，不用句号\n"
        "- 不要承诺具体时间，不要说「请稍后再试」这种客服话\n"
        "- 只输出你要说的话"
    )


FAILURE_LINES = [
    "我这边网好像抽风了，等我缓一下",
    "天气源都不理我，等会儿再问我嘛",
    "唔，查不到，网络在闹脾气",
]


def failure_template() -> str:
    return random.choice(FAILURE_LINES)

