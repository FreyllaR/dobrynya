"""Слух: слово активации, границы фраз, распознавание. Фразы озвучиваем Piper'ом.

Помечены как «models» — нужны скачанные Vosk, Silero VAD и Whisper.
"""
import json

import numpy as np
import pytest
import vosk

import config
from ears import BLOCK, Ears

pytestmark = pytest.mark.models


def _вердикт_vosk(ears_obj, pcm: np.ndarray):
    """Слова, которые Vosk услышал в записи, с уверенностью."""
    grammar = json.dumps(config.WAKE_WORDS + config.WAKE_DECOYS + ["[unk]"], ensure_ascii=False)
    rec = vosk.KaldiRecognizer(ears_obj.vosk_model, config.SAMPLE_RATE, grammar)
    rec.SetWords(True)
    data = pcm.tobytes()
    words = []
    for i in range(0, len(data), BLOCK * 2):
        if rec.AcceptWaveform(data[i:i + BLOCK * 2]):
            words += json.loads(rec.Result()).get("result", [])
    words += json.loads(rec.FinalResult()).get("result", [])
    return words


class TestСловоАктивации:
    def test_имя_узнаётся(self, ears, say):
        for фраза in ["Добрыня", "Эй, Добрыня!", "Добрыня, какая погода в Москве?"]:
            words = _вердикт_vosk(ears, say(фраза, pause=0.5))
            assert any(w["word"] in config.WAKE_WORDS and w["conf"] >= 0.85 for w in words), фраза

    def test_похожие_фразы_не_будят(self, ears, say):
        for фраза in ["Доброе утро, как дела", "Добрый вечер", "Дыня спелая", "Добро пожаловать"]:
            words = _вердикт_vosk(ears, say(фраза, pause=0.5))
            assert not any(w["word"] in config.WAKE_WORDS and w["conf"] >= 0.85 for w in words), фраза


class TestГраницыФразы:
    def test_пауза_внутри_фразы_не_обрывает(self, ears, say):
        # раньше «Напомни в девять утра ... позвонить маме» рвалось на две команды
        ears.feed(say("Напомни мне завтра в девять утра", pause=0.35), say("позвонить маме", pause=1.5))
        audio = ears.record_phrase(wait=5)
        assert audio is not None
        текст = Ears.clean_text(ears.transcribe(audio)).lower()
        # важна склейка: в команду должны попасть обе половины, до паузы и после
        assert "напомни" in текст and "позвонить" in текст, текст

    def test_тишина_возвращает_пустоту(self, ears):
        ears.feed(np.zeros(config.SAMPLE_RATE * 3, dtype=np.int16))
        assert ears.record_phrase(wait=1) is None

    def test_фон_не_глушит_следующую_фразу(self, ears, say):
        """После громкой фразы порог не должен взлетать — иначе Добрыня «глохнет» в разговоре."""
        for _ in range(60):
            ears._track_noise((np.random.default_rng(0).normal(0, 4000, BLOCK)).astype(np.int16).tobytes())
        assert ears.noise_level < 700, "после речи фон завышен — следующую реплику не услышит"
        ears.feed(say("Какая погода в Москве?"))
        assert ears.record_phrase(wait=5) is not None


class TestПолныйПуть:
    def test_имя_и_команда_одной_фразой(self, ears, say):
        ears.feed(say("Доброе утро всем", pause=0.8), say("Добрыня, какая погода в Москве?"))
        текст = Ears.clean_text(ears.transcribe(ears.wait_for_wake_word()))
        assert "погода" in текст.lower() and "добрыня" not in текст.lower(), текст

    def test_разговор_без_имени_после_ответа(self, ears, say):
        ears.feed(say("А что завтра?"))
        assert "завтра" in ears.listen(wait=5).lower()

    def test_whisper_разбирает_команды(self, ears, say):
        for фраза, слово in [("Поставь таймер на десять минут", "таймер"),
                             ("Включи музыку погромче", "музык"),
                             ("Запиши в заметки купить молоко", "молоко")]:
            текст = ears.transcribe(say(фраза, pause=0.5).astype(np.float32) / 32768)
            assert слово in текст.lower(), f"{фраза} → {текст}"
