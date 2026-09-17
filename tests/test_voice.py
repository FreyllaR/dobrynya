"""Голос: скорость, чистота звука, разборчивость, запасной голос.

Помечены как «models» — нужны скачанные Qwen3-TTS, Piper и Whisper.
"""
import difflib
import re
import time

import numpy as np
import pytest

import config
import voice as V

pytestmark = pytest.mark.models
SR = 24000


@pytest.fixture(scope="module")
def голос():
    if config.TTS_ENGINE != "qwen":
        pytest.skip("тесты написаны для голоса Qwen (TTS_ENGINE = \"qwen\")")
    v = V.Voice()
    if v.qwen is None:
        pytest.skip("Qwen3-TTS не загрузился")
    return v


def разрывы(a: np.ndarray) -> float:
    """Скачки звуковой волны в секунду — на слух это щелчки и «шлепки»."""
    d = np.abs(np.diff(a))
    local = np.convolve(d, np.ones(240) / 240, "same") + 1e-5
    return float(np.sum((d > 12 * local) & (d > 0.02)) / (len(a) / SR))


def распознано(a: np.ndarray) -> str:
    import mlx_whisper
    from scipy.signal import resample_poly
    return mlx_whisper.transcribe(resample_poly(a, 2, 3).astype(np.float32),
                                  path_or_hf_repo=config.WHISPER_MODEL, language="ru",
                                  condition_on_previous_text=False)["text"].strip()


def похожесть(услышано: str, текст: str) -> float:
    норм = lambda s: re.sub(r"[^а-яё ]", "", s.lower().replace("ё", "е"))
    return difflib.SequenceMatcher(None, норм(услышано), норм(текст)).ratio()


ФРАЗА = "В Москве семнадцать градусов и ясно, сэр. Дождя пока не обещают."


class TestКачествоГолоса:
    def test_говорит_разборчиво(self, голос):
        a = np.concatenate(list(голос._qwen_chunks(ФРАЗА)))
        assert похожесть(распознано(a), ФРАЗА) > 0.8

    def test_без_щелчков_и_шлепков(self, голос):
        a = np.concatenate(list(голос._qwen_chunks(ФРАЗА)))
        assert разрывы(a) < 0.35, "в звуке появились разрывы — проверьте ускорение речи"

    def test_звук_целый(self, голос):
        a = np.concatenate(list(голос._qwen_chunks("Готово, сэр.")))
        assert not np.isnan(a).any() and 0.05 < np.abs(a).max() <= 1.0
        assert abs(float(a.mean())) < 0.01, "постоянное смещение — будет глухой щелчок в колонках"

    def test_темп_живой_речи(self, голос):
        import torch
        from scipy.signal import resample_poly
        from silero_vad import load_silero_vad
        vad = load_silero_vad(onnx=True)
        a = np.concatenate(list(голос._qwen_chunks(ФРАЗА)))
        a16 = resample_poly(a, 2, 3).astype(np.float32)
        vad.reset_states()
        речь = sum(vad(torch.from_numpy(a16[i:i + 512]), 16000).item() > .5
                   for i in range(0, len(a16) - 512, 512)) * 512 / 16000
        слогов = len(re.findall(r"[аеёиоуыэюяАЕЁИОУЫЭЮЯ]", ФРАЗА))
        assert 4.0 < слогов / речь < 7.5, f"{слогов / речь:.1f} слогов/с — слишком медленно или тараторит"

    def test_ускорение_укорачивает_речь(self, голос, monkeypatch):
        monkeypatch.setattr(config, "QWEN_SPEED", 1.0)
        обычно = sum(len(c) for c in голос._qwen_chunks(ФРАЗА)) / SR
        monkeypatch.setattr(config, "QWEN_SPEED", 1.2)
        быстро = sum(len(c) for c in голос._qwen_chunks(ФРАЗА)) / SR
        assert 0.7 < быстро / обычно < 0.95, f"{обычно:.1f} с → {быстро:.1f} с"


class TestПотокИЗапасныеГолоса:
    def test_первый_звук_быстро_и_без_пауз(self, голос, monkeypatch):
        паузы, часы = [], {"t": None}

        class Колонки:
            def __init__(self, samplerate, channels, dtype): self.sr = samplerate
            def __enter__(self): return self
            def __exit__(self, *a): pass
            def write(self, a):
                now = time.time()
                if часы["t"] is None:
                    часы["t"] = now
                elif now > часы["t"] + 0.02:
                    паузы.append(now - часы["t"]); часы["t"] = now
                часы["t"] += len(a) / self.sr
                time.sleep(max(0, часы["t"] - time.time() - 0.05))

        import sounddevice as sd
        monkeypatch.setattr(sd, "OutputStream", Колонки)
        начало = time.time()
        первый = {}
        голос.say_stream(iter([ФРАЗА + " И ещё одно предложение для проверки."]),
                         on_start=lambda: первый.setdefault("t", time.time() - начало))
        assert первый["t"] < 3.0, "слишком долго до первого звука"
        assert not паузы, f"звук рвался: {паузы}"

    def test_запасной_голос_piper(self, голос):
        assert голос.piper is not None, "Piper должен быть готов на случай сбоя Qwen"
        a = голос._piper("Проверка запасного голоса, сэр.")
        assert len(a) / 48000 > 1.0 and np.abs(a).max() > 0.05

    def test_при_сбое_модели_говорит_piper(self, голос, monkeypatch):
        сыграно = []

        class Колонки:
            def __init__(self, samplerate, channels, dtype): pass
            def __enter__(self): return self
            def __exit__(self, *a): pass
            def write(self, a): сыграно.append(len(a))

        import sounddevice as sd
        monkeypatch.setattr(sd, "OutputStream", Колонки)
        monkeypatch.setattr(голос, "_qwen_chunks", lambda text: (_ for _ in ()).throw(RuntimeError("сбой модели")))
        голос.say_stream(iter(["Проверка, сэр."]))
        assert сыграно, "при сбое Qwen Добрыня должен был договорить запасным голосом"


class TestОбразцыГолоса:
    def test_оба_образца_на_месте(self):
        for имя, (path, text) in config.QWEN_VOICES.items():
            assert path.exists(), f"нет образца голоса {имя}: {path}"
            assert len(text) > 20, f"у образца {имя} нет текста — клонирование сломается"
