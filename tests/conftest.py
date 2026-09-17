"""Общая обвязка тестов: изоляция данных, метки, звуковые заготовки."""
import collections
import json
import sys
import time
import wave
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as dobrynya  # noqa: E402  (имя config занято самим pytest)

FIXTURES = dobrynya.DATA_DIR / "test_fixtures"


def pytest_configure(config):
    config.addinivalue_line("markers", "models: нужны скачанные модели (медленно)")
    config.addinivalue_line("markers", "net: обращается к Gemini и тратит бесплатный лимит")


def pytest_addoption(parser):
    parser.addoption("--net", action="store_true", help="включить тесты, которые обращаются к Gemini")
    parser.addoption("--quick", action="store_true", help="только быстрые тесты, без моделей")


def pytest_collection_modifyitems(config, items):
    skip_net = pytest.mark.skip(reason="нужен флаг --net (тратит лимит Gemini)")
    skip_models = pytest.mark.skip(reason="режим --quick: тесты с моделями пропущены")
    for item in items:
        if "net" in item.keywords and not config.getoption("--net"):
            item.add_marker(skip_net)
        if "models" in item.keywords and config.getoption("--quick"):
            item.add_marker(skip_models)


@pytest.fixture
def isolated_data(tmp_path, monkeypatch):
    """Заметки, напоминания и память — во временной папке, данные пользователя не трогаем."""
    import tools
    for name in ("NOTES_FILE", "REMINDERS_FILE", "MEMORY_FILE"):
        monkeypatch.setattr(tools, name, tmp_path / f"{name.lower()}.json")
    return tmp_path


@pytest.fixture(scope="session")
def piper():
    """Piper — быстрый локальный голос, им озвучиваем фразы для проверки «ушей»."""
    path = dobrynya.BASE_DIR / "models" / "piper" / f"{dobrynya.PIPER_VOICE}.onnx"
    if not path.exists():
        pytest.skip("нет модели Piper — запустите ./run.sh, он их скачает")
    from piper import PiperVoice
    return PiperVoice.load(path)


@pytest.fixture(scope="session")
def say(piper):
    """Озвучивает фразу и отдаёт 16 кГц моно; результат кэшируется в data/test_fixtures."""
    from scipy.signal import resample_poly

    FIXTURES.mkdir(exist_ok=True)

    def _say(text: str, pause: float = 1.5) -> np.ndarray:
        path = FIXTURES / (f"{abs(hash(text)) % 10**10}.wav")
        if path.exists():
            with wave.open(str(path)) as w:
                pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        else:
            chunks = [c.audio_float_array for c in piper.synthesize(text)]
            audio = resample_poly(np.concatenate(chunks), dobrynya.SAMPLE_RATE, piper.config.sample_rate)
            pcm = (np.clip(audio, -1, 1) * 32767 * 0.6).astype(np.int16)
            with wave.open(str(path), "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(dobrynya.SAMPLE_RATE)
                w.writeframes(pcm.tobytes())
        return np.concatenate([pcm, np.zeros(int(pause * dobrynya.SAMPLE_RATE), dtype=np.int16)])

    return _say


@pytest.fixture
def ears(monkeypatch):
    """«Уши» без микрофона: звук подаём сами через очередь."""
    import queue
    import vosk
    import ears as ears_module

    if not dobrynya.VOSK_MODEL_PATH.exists():
        pytest.skip("нет модели Vosk — запустите ./run.sh, он её скачает")
    vosk.SetLogLevel(-1)
    class МикрофонИзФайла:
        """Ведёт себя как живой микрофон: звук идёт сплошной лентой, а когда запись кончилась —
        отдаёт тишину. Сброс буфера (flush) ленту не стирает, как и в жизни."""
        def __init__(self):
            self.tape = collections.deque()

        def put(self, data):
            self.tape.append(data)

        def empty(self):
            return True  # Добрыне нечего выбрасывать: микрофон пишет дальше

        def get_nowait(self):
            raise queue.Empty

        def get(self, *a, **k):
            if not self.tape:
                time.sleep(0.01)  # чтобы счётчики времени в «ушах» шли как в жизни
                return bytes(ears_module.BLOCK * 2)
            return self.tape.popleft()

    e = ears_module.Ears.__new__(ears_module.Ears)
    e.vosk_model = vosk.Model(str(dobrynya.VOSK_MODEL_PATH))
    e.q = МикрофонИзФайла()
    e.noise_level = 300.0
    e.speech_ended = 0.0

    def feed(*pcm_parts: np.ndarray):
        data = np.concatenate(pcm_parts).tobytes()
        for i in range(0, len(data), ears_module.BLOCK * 2):
            e.q.put(data[i:i + ears_module.BLOCK * 2])

    e.feed = feed
    return e


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
