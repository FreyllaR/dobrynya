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
USER_NAME = "мистер Ольков"       # как Добрыня знает тебя (ударение для голоса — в NAME_PRONUNCIATION)
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
# Какой микрофон слушать: часть названия устройства или None — системный по умолчанию.
# Встроенный микрофон лучше Bluetooth-гарнитуры: когда программа берёт микрофон гарнитуры,
# macOS переводит её в «телефонный» режим и голос Добрыни в наушниках начинает хрипеть и рваться.
MIC_DEVICE = "MacBook"
SILENCE_SECONDS = 1.0             # пауза, после которой считаем, что ты договорил
MAX_RECORD_SECONDS = 20
WAIT_SPEECH_SECONDS = 6           # сколько ждать начала фразы после «Добрыня»
CONVERSATION_TIMEOUT = 8          # сколько секунд после ответа слушать без слова «Добрыня»
SHOW_ORB = True                   # окно с анимированным шаром (Esc — закрыть окно)
ORB_ON_TOP = False                # True — окно всегда поверх остальных
SHOW_TIMINGS = True               # печатать, сколько заняло каждое звено

def _models_downloaded(*repos: str) -> bool:
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    return all(any((hub / f"models--{r.replace('/', '--')}" / "snapshots").glob("*/*")) for r in repos)


# --- Голос ---
TTS_ENGINE = "qwen"               # "qwen" — живой низкий голос богатыря, локально (~0,3 с до звука);
                                  # "piper" — мгновенно, но проще; "gemini" — облако, лимит 10 ответов в день;
                                  # "edge", "silero", "say" — запасные

QWEN_MODEL = "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit"  # точнее, но тяжелее: ...-1.7B-Base-4bit
# Образец голоса: Добрыня копирует тембр и манеру. Оба голоса сгенерированы нейросетью.
QWEN_VOICES = {
    "baritone": (BASE_DIR / "voices" / "dobrynya.wav",
                 "Добрый вечер, сэр. Все системы работают в штатном режиме. В Москве сейчас семнадцать градусов и ясно, дождя не обещают."),
    "bogatyr": (BASE_DIR / "voices" / "dobrynya_bogatyr.wav",
                "Добрый вечер, сэр. Хм... все системы работают в штатном режиме. Напомнить вам позвонить маме в девять утра?"),
}
QWEN_VOICE_NAME = "baritone"      # "baritone" — ровные интонации; "bogatyr" — прежний, басовитее, но чаще срывается
QWEN_VOICE, QWEN_VOICE_TEXT = QWEN_VOICES[QWEN_VOICE_NAME]
QWEN_VOLUME = 0.9
QWEN_SPEED = 1.2                  # темп речи: 1.0 — как генерирует модель (медленно), 1.2 — обычный разговор
QWEN_TEMPERATURE = 0.6            # меньше — ровнее интонации, больше — живее, но чаще странные ударения

GEMINI_TTS_MODEL = "gemini-3.1-flash-tts-preview"
GEMINI_VOICE = "Charon"           # мужские: Charon, Algieba, Sadaltager, Iapetus, Orus, Enceladus
GEMINI_VOICE_STYLE = (            # режиссёрская заметка: как именно говорить
    "Прочитай по-русски голосом невозмутимого, интеллигентного ИИ-дворецкого: спокойно, бархатно, "
    "негромко, в размеренном темпе, с едва заметной иронией и тёплыми живыми интонациями. "
    "Теги в квадратных скобках — это звуки (вздох, смешок), их не произноси словами."
)
GEMINI_VOLUME = 0.9
GEMINI_TTS_TIMEOUT = 8            # секунд ждать первый звук, потом — запасной голос Piper
VOICE_NONVERBAL = False           # True — вздохи, смешки и «хм» в ответах (для gemini и qwen)

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

# Как голос Qwen должен читать слова, где модель ошибается с ударением: слово → написание «для слуха».
# Qwen не понимает знаки ударения, поэтому подбираем написание, которое звучит правильно.
NAME_PRONUNCIATION = {}           # пример: {"Ольков": "Олькофф"}; «Ольков» модель и так читает с ударением на последний слог

# Если Добрыня ставит ударение неправильно — допиши слово сюда.
# «+» ставится ПЕРЕД ударной гласной. Регистр первой буквы сохранится сам.
STRESS_FIXES = {
    "добрыня": "добр+ыня",
    "добрыню": "добр+ыню",
    "добрыне": "добр+ыне",
}
SAY_VOICE = "Milena"              # голос для движка "say"

# Модели уже скачаны — не проверяем обновления на HuggingFace при каждом запуске:
# с медленной сетью или VPN эта проверка растягивала загрузку на десятки секунд.
if _models_downloaded(WHISPER_MODEL, QWEN_MODEL):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
