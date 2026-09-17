"""Мозг (без обращения к сети) и окно с шаром."""
import json
import urllib.request

import numpy as np
import pytest

import brain
import config
import tools
import ui


class TestХарактер:
    def test_имя_и_обращение(self, monkeypatch):
        monkeypatch.setattr(tools, "get_memory", lambda: [])
        prompt = brain.system_prompt()
        assert config.NAME in prompt and "сэр" in prompt
        assert config.USER_NAME in prompt if config.USER_NAME else True
        assert "Отвечай по-русски" in prompt

    def test_междометия_запрещены_когда_выключены(self, monkeypatch):
        monkeypatch.setattr(tools, "get_memory", lambda: [])
        monkeypatch.setattr(config, "VOICE_NONVERBAL", False)
        assert "Никаких междометий" in brain.system_prompt()
        monkeypatch.setattr(config, "VOICE_NONVERBAL", True)
        monkeypatch.setattr(config, "TTS_ENGINE", "qwen")
        assert "[sighs]" in brain.system_prompt()

    def test_память_попадает_в_характер(self, monkeypatch):
        monkeypatch.setattr(tools, "get_memory", lambda: ["Пользователь любит джаз"])
        assert "джаз" in brain.system_prompt()


class TestИсторияРазговора:
    def test_обрезается_по_границе_реплики(self, monkeypatch):
        from google.genai import types
        b = brain.GeminiBrain.__new__(brain.GeminiBrain)
        реплика = lambda role, text: types.Content(role=role, parts=[types.Part(text=text)])
        вызов = types.Content(role="model", parts=[types.Part(
            function_call=types.FunctionCall(name="get_weather", args={}))])
        история = [реплика("user", "погода"), вызов, types.Content(
            role="user", parts=[types.Part(function_response=types.FunctionResponse(name="get_weather", response={}))]),
            реплика("model", "дождь")] * 8
        monkeypatch.setattr(config, "HISTORY_LIMIT", 20)
        обрезанная = b._trim(история)
        assert len(обрезанная) <= 20
        assert обрезанная[0].role == "user" and обрезанная[0].parts[0].text, "история должна начинаться с реплики человека"

    def test_куски_ответа_склеиваются(self):
        from google.genai import types
        b = brain.GeminiBrain.__new__(brain.GeminiBrain)
        куски = [types.Content(role="user", parts=[types.Part(text="привет")]),
                 types.Content(role="model", parts=[types.Part(text="При")]),
                 types.Content(role="model", parts=[types.Part(text="вет, сэр")])]
        склеено = b._compact(куски)
        assert len(склеено) == 2 and склеено[1].parts[0].text == "Привет, сэр"


class TestШар:
    def test_состояния_и_громкость(self):
        ui.set_state("listening", user="какая погода", answer="")
        ui.mic_level(3000)
        snap = ui._snapshot()
        assert snap["state"] == "listening" and snap["user"] == "какая погода"
        assert 0.9 <= snap["level"] <= 1.0

    def test_рот_двигается_в_такт_речи(self):
        sr = 24000
        t = np.arange(sr) / sr
        громко = (np.sin(2 * np.pi * 120 * t) * np.abs(np.sin(2 * np.pi * 3 * t))).astype(np.float32)
        ui.speaking(громко, sr)
        snap = ui._snapshot()
        assert snap["state"] == "speaking"
        assert 0.0 <= snap["level"] <= 1.0
        ui.speaking_done()
        assert ui._snapshot()["state"] != "speaking"

    def test_сервер_отдаёт_страницу_и_состояние(self, monkeypatch):
        launched = []
        monkeypatch.setattr(ui.subprocess, "Popen", lambda *a, **k: launched.append(a) or type(
            "P", (), {"terminate": lambda self: None})())
        ui.start()
        port = launched and None
        # сервер поднят в ui.start(); находим его порт по последнему аргументу командной строки окна
        url = [x for x in launched[0][0] if isinstance(x, str) and x.startswith("http")][0]
        page = urllib.request.urlopen(url, timeout=5).read().decode()
        assert "<svg" in page or "canvas" in page
        state = json.loads(urllib.request.urlopen(url + "state", timeout=5).read().decode())
        assert {"state", "level", "user", "answer"} <= set(state)


@pytest.mark.net
class TestРазговорСGemini:
    def test_отвечает_по_русски_и_зовёт_навык(self, monkeypatch, isolated_data):
        monkeypatch.setattr(tools, "get_memory", lambda: [])
        вызвано = []
        b = brain.Brain()
        ответ = b.ask("Какая погода в Казани?", on_tool=lambda name, args: вызвано.append(name))
        assert "get_weather" in вызвано, "Gemini должен был вызвать навык погоды"
        assert any(c.isalpha() and c.lower() in "абвгдеёжзийклмнопрстуфхцчшщъыьэюя" for c in ответ)
        assert len(ответ) < 400, "ответ должен быть коротким — его слушают"

    def test_помнит_контекст(self, monkeypatch, isolated_data):
        monkeypatch.setattr(tools, "get_memory", lambda: [])
        b = brain.Brain()
        b.ask("Меня зовут Кирилл, запомни на время разговора.")
        assert "кирилл" in b.ask("Как меня зовут?").lower()
