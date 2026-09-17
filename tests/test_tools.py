"""Навыки: заметки, напоминания, память, погода, описания для Gemini."""
import inspect
import json
import time
from datetime import datetime, timedelta

import pytest

import tools


class TestЗаметкиИПамять:
    def test_заметка_сохраняется_и_читается(self, isolated_data):
        tools.add_note("купить молоко")
        tools.add_note("позвонить в сервис")
        notes = tools.read_notes()
        assert "купить молоко" in notes and "позвонить в сервис" in notes
        assert tools.clear_notes() and tools.read_notes() == "Заметок нет"

    def test_память_переживает_перезапуск(self, isolated_data):
        tools.remember_fact("Пользователь любит джаз")
        assert tools.get_memory() == ["Пользователь любит джаз"]
        assert "джаз" in json.loads(tools.MEMORY_FILE.read_text(encoding="utf-8"))[0]


class TestНапоминания:
    def test_через_минуты(self, isolated_data):
        answer = tools.set_reminder("выпить воды", in_minutes=30)
        assert "выпить воды" in answer
        assert "выпить воды" in tools.list_reminders()

    def test_на_время_переносится_на_завтра_если_время_прошло(self, isolated_data):
        past = (datetime.now() - timedelta(hours=1)).strftime("%H:%M")
        answer = tools.set_reminder("позвонить маме", at_time=past)
        when = datetime.fromtimestamp(json.loads(tools.REMINDERS_FILE.read_text(encoding="utf-8"))[0]["at"])
        assert when > datetime.now(), answer

    def test_отмена(self, isolated_data):
        tools.set_reminder("зарядить наушники", in_minutes=5)
        tools.set_reminder("вынести мусор", in_minutes=10)
        tools.cancel_reminder("мусор")
        assert "наушники" in tools.list_reminders() and "мусор" not in tools.list_reminders()
        tools.cancel_reminder("все")
        assert tools.list_reminders() == "Напоминаний нет"

    def test_срабатывает_и_говорит_вслух(self, isolated_data, monkeypatch):
        сказано = []
        monkeypatch.setattr(tools, "_osascript", lambda script: "")
        tools.set_notifier(сказано.append)
        try:
            tools.set_reminder("проверка", in_minutes=1 / 60)  # через секунду
            for _ in range(40):
                time.sleep(0.1)
                if сказано:
                    break
        finally:
            tools.set_notifier(print)
        assert сказано and "проверка" in сказано[0]


class TestПогода:
    def test_город_в_падеже_понимается(self, monkeypatch):
        запрошено = []

        class Ответ:
            def __init__(self, data): self.data = data
            def json(self): return self.data

        def fake_get(url, params=None, timeout=None):
            if "geocoding" in url:
                запрошено.append(params["name"])
                if params["name"] != "Казань":
                    return Ответ({})
                return Ответ({"results": [{"name": "Казань", "latitude": 55.8, "longitude": 49.1}]})
            return Ответ({"current": {"temperature_2m": 14, "apparent_temperature": 13, "weather_code": 63,
                                      "wind_speed_10m": 3},
                          "daily": {"temperature_2m_max": [14, 16, 18], "temperature_2m_min": [12, 10, 9],
                                    "weather_code": [63, 3, 3], "precipitation_probability_max": [95, 3, 0]}})

        monkeypatch.setattr(tools.requests, "get", fake_get)
        answer = tools.get_weather("Казани")
        assert "Казань" in answer and "14" in answer and "дождь" in answer
        assert запрошено[0] == "Казани" and "Казань" in запрошено

    def test_неизвестный_город_не_роняет(self, monkeypatch):
        monkeypatch.setattr(tools.requests, "get", lambda *a, **k: type("R", (), {"json": lambda self: {}})())
        assert "не нашёл" in tools.get_weather("Бурляндия").lower()


class TestОписанияДляGemini:
    def test_у_каждого_навыка_есть_описание_и_типы(self):
        for fn in tools.TOOLS:
            assert fn.__doc__, f"{fn.__name__}: нет docstring — Gemini не поймёт, зачем навык"
            sig = inspect.signature(fn)
            for name, p in sig.parameters.items():
                assert p.annotation is not inspect.Parameter.empty, f"{fn.__name__}: у «{name}» нет типа"
                if p.default is inspect.Parameter.empty:
                    continue
                assert f"{name}:" in fn.__doc__, f"{fn.__name__}: «{name}» не описан в docstring"

    def test_список_и_карта_совпадают(self):
        assert set(tools.TOOL_MAP) == {fn.__name__ for fn in tools.TOOLS}
        assert len(tools.TOOLS) >= 15

    @pytest.mark.parametrize("fn", [t for t in tools.TOOLS if t.__name__ in
                                    ("get_datetime", "list_reminders", "read_notes")])
    def test_безопасные_навыки_работают_без_аргументов(self, fn, isolated_data):
        assert isinstance(fn(), str)
