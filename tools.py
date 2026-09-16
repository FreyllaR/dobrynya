"""Навыки Добрыни. Каждая функция — инструмент, который может вызвать LLM.

Чтобы добавить свой навык: напиши функцию с понятным docstring и
аннотациями типов, затем добавь её в список TOOLS внизу файла.
"""
import json
import subprocess
import threading
import time
import urllib.parse
import uuid
from datetime import datetime, timedelta

import requests

import config

NOTES_FILE = config.DATA_DIR / "notes.json"
REMINDERS_FILE = config.DATA_DIR / "reminders.json"
MEMORY_FILE = config.DATA_DIR / "memory.json"

_lock = threading.Lock()
_notify = print  # заменяется в main.py на «сказать вслух»


def set_notifier(fn):
    global _notify
    _notify = fn


def _load(path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _osascript(script: str) -> str:
    r = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
    return (r.stdout or r.stderr).strip()


# ---------------------------------------------------------------- Время

def get_datetime() -> str:
    """Возвращает текущие дату, время и день недели."""
    days = ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"]
    now = datetime.now()
    return f"{now:%d.%m.%Y %H:%M}, {days[now.weekday()]}"


# ---------------------------------------------------------------- Компьютер

def open_app(name: str) -> str:
    """Открывает приложение на Mac.

    Args:
        name: название приложения на английском, как в папке Applications, например "Safari", "Telegram", "Music"
    """
    r = subprocess.run(["open", "-a", name], capture_output=True, text=True)
    return f"Открыл {name}" if r.returncode == 0 else f"Не нашёл приложение {name}"


def open_website(url_or_query: str) -> str:
    """Открывает сайт в браузере. Если передан не адрес, а запрос — открывает поиск.

    Args:
        url_or_query: адрес сайта (youtube.com) или поисковый запрос
    """
    q = url_or_query.strip()
    if " " not in q and "." in q:
        url = q if q.startswith("http") else "https://" + q
    else:
        url = "https://yandex.ru/search/?text=" + urllib.parse.quote(q)
    subprocess.run(["open", url])
    return f"Открыл {url}"


def set_volume(level: int) -> str:
    """Устанавливает громкость звука компьютера.

    Args:
        level: громкость от 0 до 100
    """
    level = max(0, min(100, int(level)))
    _osascript(f"set volume output volume {level}")
    return f"Громкость {level}"


def media_control(action: str) -> str:
    """Управляет музыкой в Spotify или Apple Music.

    Args:
        action: одно из: play, pause, next, previous
    """
    commands = {"play": "play", "pause": "pause", "next": "next track", "previous": "previous track"}
    if action not in commands:
        return "Неизвестное действие"
    running = _osascript('tell application "System Events" to (name of processes) contains "Spotify"')
    app = "Spotify" if running == "true" else "Music"
    _osascript(f'tell application "{app}" to {commands[action]}')
    return f"{app}: {action}"


def find_files(query: str) -> str:
    """Ищет файлы на компьютере по имени.

    Args:
        query: часть имени файла
    """
    r = subprocess.run(["mdfind", "-onlyin", str(config.BASE_DIR.home()), "-name", query],
                       capture_output=True, text=True, timeout=15)
    files = [f for f in r.stdout.splitlines() if "/Library/" not in f][:10]
    return "\n".join(files) if files else "Ничего не нашёл"


def open_file(path: str) -> str:
    """Открывает файл или папку по полному пути (например найденный через find_files).

    Args:
        path: полный путь к файлу
    """
    r = subprocess.run(["open", path], capture_output=True, text=True)
    return "Открыл" if r.returncode == 0 else "Не получилось открыть"


# ---------------------------------------------------------------- Напоминания

def _reminder_loop():
    while True:
        time.sleep(1)
        now = time.time()
        with _lock:
            items = _load(REMINDERS_FILE, [])
            due = [r for r in items if r["at"] <= now]
            if due:
                _save(REMINDERS_FILE, [r for r in items if r["at"] > now])
        for r in due:
            text = r["text"]
            _osascript(f'display notification "{text}" with title "{config.NAME}" sound name "Glass"')
            _notify(f"Сэр, напоминаю: {text}")


threading.Thread(target=_reminder_loop, daemon=True).start()


def set_reminder(text: str, in_minutes: float = 0, at_time: str = "") -> str:
    """Ставит напоминание или таймер. Укажи либо in_minutes, либо at_time.

    Args:
        text: о чём напомнить (для таймера — "таймер")
        in_minutes: через сколько минут напомнить (можно дробное, 0.5 = 30 секунд)
        at_time: время в формате ЧЧ:ММ, например "18:30"
    """
    now = datetime.now()
    if at_time:
        h, m = map(int, at_time.split(":"))
        when = now.replace(hour=h, minute=m, second=0, microsecond=0)
        if when <= now:
            when += timedelta(days=1)
    elif in_minutes:
        when = now + timedelta(minutes=float(in_minutes))
    else:
        return "Не понял, когда напомнить"
    with _lock:
        items = _load(REMINDERS_FILE, [])
        items.append({"id": uuid.uuid4().hex[:6], "text": text, "at": when.timestamp()})
        _save(REMINDERS_FILE, items)
    return f"Напомню «{text}» в {when:%H:%M:%S} ({when:%d.%m})"


def list_reminders() -> str:
    """Показывает все активные напоминания и таймеры."""
    items = sorted(_load(REMINDERS_FILE, []), key=lambda r: r["at"])
    if not items:
        return "Напоминаний нет"
    return "\n".join(f"{datetime.fromtimestamp(r['at']):%d.%m %H:%M} — {r['text']}" for r in items)


def cancel_reminder(text: str) -> str:
    """Отменяет напоминания, в тексте которых есть указанные слова.

    Args:
        text: часть текста напоминания; "все" — отменить все
    """
    with _lock:
        items = _load(REMINDERS_FILE, [])
        keep = [] if text.lower() in ("все", "всё") else [r for r in items if text.lower() not in r["text"].lower()]
        _save(REMINDERS_FILE, keep)
    return f"Отменено: {len(items) - len(keep)}"


# ---------------------------------------------------------------- Заметки и память

def add_note(text: str) -> str:
    """Сохраняет заметку (список покупок, идея, мысль).

    Args:
        text: текст заметки
    """
    notes = _load(NOTES_FILE, [])
    notes.append({"date": f"{datetime.now():%d.%m %H:%M}", "text": text})
    _save(NOTES_FILE, notes)
    return "Записал"


def read_notes() -> str:
    """Читает все сохранённые заметки."""
    notes = _load(NOTES_FILE, [])
    return "\n".join(f"{n['date']}: {n['text']}" for n in notes) or "Заметок нет"


def clear_notes() -> str:
    """Удаляет все заметки."""
    _save(NOTES_FILE, [])
    return "Заметки очищены"


def remember_fact(fact: str) -> str:
    """Запоминает навсегда важный факт о пользователе (имя, предпочтения, дни рождения близких и т.п.).

    Args:
        fact: факт одним предложением
    """
    facts = _load(MEMORY_FILE, [])
    facts.append(fact)
    _save(MEMORY_FILE, facts)
    return "Запомнил"


def get_memory() -> list[str]:
    return _load(MEMORY_FILE, [])


# ---------------------------------------------------------------- Интернет

WEATHER_CODES = {
    0: "ясно", 1: "преимущественно ясно", 2: "переменная облачность", 3: "пасмурно",
    45: "туман", 48: "изморозь", 51: "лёгкая морось", 53: "морось", 55: "сильная морось",
    61: "небольшой дождь", 63: "дождь", 65: "сильный дождь", 66: "ледяной дождь", 67: "ледяной дождь",
    71: "небольшой снег", 73: "снег", 75: "сильный снег", 77: "снежная крупа",
    80: "ливень", 81: "ливень", 82: "сильный ливень", 85: "снегопад", 86: "сильный снегопад",
    95: "гроза", 96: "гроза с градом", 99: "гроза с градом",
}


def _find_city(city: str):
    # модель часто передаёт город в падеже: «Казани», «Москве», «Петербурге»
    variants = [city]
    for end, repl in (("и", "ь"), ("и", "а"), ("е", "а"), ("е", ""), ("у", "а"), ("ы", "а")):
        if city.endswith(end):
            variants.append(city[: -len(end)] + repl)
    for name in variants:
        geo = requests.get("https://geocoding-api.open-meteo.com/v1/search",
                           params={"name": name, "count": 1, "language": "ru"}, timeout=10).json()
        if geo.get("results"):
            return geo["results"][0]
    return None


def get_weather(city: str = "") -> str:
    """Погода сейчас и прогноз на 3 дня.

    Args:
        city: город в именительном падеже (Казань, а не Казани); если не указан — город пользователя по умолчанию
    """
    place = _find_city(city or config.DEFAULT_CITY)
    if not place:
        return f"Не нашёл город {city}"
    w = requests.get("https://api.open-meteo.com/v1/forecast", params={
        "latitude": place["latitude"], "longitude": place["longitude"],
        "current": "temperature_2m,apparent_temperature,weather_code,wind_speed_10m",
        "daily": "temperature_2m_max,temperature_2m_min,weather_code,precipitation_probability_max",
        "timezone": "auto", "forecast_days": 3, "wind_speed_unit": "ms",
    }, timeout=10).json()
    c, d = w["current"], w["daily"]
    lines = [f"{place['name']} сейчас: {round(c['temperature_2m'])}°, ощущается как "
             f"{round(c['apparent_temperature'])}°, {WEATHER_CODES.get(c['weather_code'], '')}, "
             f"ветер {round(c['wind_speed_10m'])} м/с"]
    for i, day in enumerate(["сегодня", "завтра", "послезавтра"]):
        lines.append(f"{day}: от {round(d['temperature_2m_min'][i])}° до {round(d['temperature_2m_max'][i])}°, "
                     f"{WEATHER_CODES.get(d['weather_code'][i], '')}, осадки {d['precipitation_probability_max'][i]}%")
    return "\n".join(lines)


def web_search(query: str) -> str:
    """Ищет актуальную информацию в интернете. Используй, когда не знаешь ответа или нужны свежие данные.

    Args:
        query: поисковый запрос
    """
    from ddgs import DDGS
    results = DDGS().text(query, region="ru-ru", max_results=5)
    return "\n\n".join(f"{r['title']}: {r['body']}" for r in results) or "Ничего не нашёл"


def get_news(topic: str = "") -> str:
    """Свежие новости, общие или по теме.

    Args:
        topic: тема новостей (необязательно)
    """
    if topic:
        from ddgs import DDGS
        results = DDGS().news(topic, region="ru-ru", max_results=6)
        return "\n".join(f"- {r['title']}" for r in results) or "Новостей не нашёл"
    import feedparser
    feed = feedparser.parse(requests.get("https://lenta.ru/rss/top7", timeout=10).content)
    return "\n".join(f"- {e.title}" for e in feed.entries[:7]) or "Новостей не нашёл"


TOOLS = [
    get_datetime, open_app, open_website, set_volume, media_control, find_files, open_file,
    set_reminder, list_reminders, cancel_reminder, add_note, read_notes, clear_notes,
    remember_fact, get_weather, web_search, get_news,
]
TOOL_MAP = {f.__name__: f for f in TOOLS}
