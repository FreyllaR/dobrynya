"""Настройки Добрыни. Меняй под себя."""
import os
from pathlib import Path

import certifi

# python.org-сборка Python не видит системные сертификаты — даём ей certifi
os.environ.setdefault("SSL_CERT_FILE", certifi.where())

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# --- Личность ---
NAME = "Добрыня"
WAKE_WORDS = ["добрыня"]          # как Vosk слышит имя (можно добавить варианты)
# похожие по звучанию слова, чтобы «Доброе утро» не будило Добрыню
WAKE_DECOYS = ["добрый", "доброе", "добро", "доброй", "добрая", "дыня", "доня",
               "утро", "день", "вечер", "ночи", "дорогая", "добрался"]
USER_NAME = ""                    # как обращаться к тебе, например "Кирилл"
DEFAULT_CITY = "Москва"           # для погоды по умолчанию

# --- Мозг ---
LLM_PROVIDER = "gemini"           # "gemini" (облако, бесплатный лимит) или "ollama" (локально)
# модели по порядку: если первая недоступна или кончился лимит — берётся следующая.
# lite отвечает за 1–2 секунды; flash умнее, но на бесплатном тарифе часто перегружены (10–25 с)
GEMINI_MODELS = ["gemini-3.5-flash-lite", "gemini-3.5-flash", "gemini-3.8-flash"]
GEMINI_TIMEOUT = 15               # секунд на ответ, потом пробуем следующую модель
OLLAMA_MODEL = "qwen3:8b"


def _read_env(path: Path) -> dict:
    if not path.exists():
        return {}
    pairs = (line.split("=", 1) for line in path.read_text().splitlines() if "=" in line and not line.startswith("#"))
    return {k.strip(): v.strip().strip('"') for k, v in pairs}


GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or _read_env(BASE_DIR / ".env").get("GEMINI_API_KEY", "")
HISTORY_LIMIT = 20                # сколько последних сообщений помнить в разговоре

# --- Уши ---
VOSK_MODEL_PATH = BASE_DIR / "models" / "vosk-model-small-ru-0.22"
WHISPER_MODEL = "mlx-community/whisper-large-v3-turbo"
SAMPLE_RATE = 16000
SILENCE_SECONDS = 0.8             # пауза, после которой считаем, что ты договорил
MAX_RECORD_SECONDS = 20
WAIT_SPEECH_SECONDS = 6           # сколько ждать начала фразы после «Добрыня»
CONVERSATION_TIMEOUT = 8          # сколько секунд после ответа слушать без слова «Добрыня»
SHOW_ORB = True                   # окно с анимированным шаром (Esc — закрыть окно)
SHOW_TIMINGS = True               # печатать, сколько заняло каждое звено

# --- Голос ---
TTS_ENGINE = "piper"              # "piper" — мгновенно и локально; "edge" — Dmitry из облака (задержка 1–5 с);
                                  # "silero" — офлайн; "say" — встроенный macOS
VOICE_EFFECT = "soft"             # "soft" — мягкий тёплый тембр; "" — без обработки
VOICE_SOFTNESS = 0.7              # 0 — почти без обработки, 1 — максимально бархатно
VOICE_VOLUME = 0.6                # громкость голоса, 0.1–1.0

PIPER_VOICE = "ru_RU-ruslan-medium"  # самый низкий; ещё: ru_RU-denis-medium, ru_RU-dmitri-medium (молодой)
PIPER_PITCH = 0.86                # тон и тембр: 1.0 — как есть, 0.84 — басовитее, < 0.8 — уже «великан»
PIPER_SPEED = 1.15                # > 1 — медленнее и спокойнее
PIPER_NOISE = 0.35                # «зернистость» и живость интонации: меньше — ровнее и спокойнее
PIPER_NOISE_W = 0.45              # неровность темпа: меньше — плавнее

EDGE_VOICE = "ru-RU-DmitryNeural"
EDGE_RATE = "-8%"                 # темп: "-12%" — спокойнее, "+0%" — обычный
EDGE_PITCH = "-6Hz"               # тон: "-12Hz" — ниже, "+0Hz" — как есть

# запасной офлайн-голос, если пропал интернет
SILERO_MODEL = "v5_5_ru"          # 5-я версия: сама ставит ударения, «ё» и вопросительную интонацию
SILERO_SPEAKER = "eugene"         # мужские: aidar, eugene; женские: baya, kseniya, xenia
SILERO_PITCH = "low"              # x-low, low, medium, high, x-high
SILERO_RATE = "slow"              # x-slow, slow, medium, fast, x-fast
SAY_VOICE = "Milena"              # голос для движка "say"

# Если Добрыня ставит ударение неправильно — допиши слово сюда.
# «+» ставится ПЕРЕД ударной гласной. Регистр первой буквы сохранится сам.
STRESS_FIXES = {
    "добрыня": "добр+ыня",
    "добрыню": "добр+ыню",
    "добрыне": "добр+ыне",
}
SAY_VOICE = "Milena"              # голос для движка "say"
