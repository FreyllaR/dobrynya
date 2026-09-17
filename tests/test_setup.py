"""Сборка и настройки: всё на месте, ничего не сломано до запуска."""
import compileall
import os
import subprocess
import sys

import config


class TestФайлыПроекта:
    def test_код_компилируется(self):
        assert compileall.compile_dir(str(config.BASE_DIR), quiet=2, maxlevels=0), "есть синтаксические ошибки"

    def test_скрипт_запуска_корректен(self):
        assert subprocess.run(["bash", "-n", str(config.BASE_DIR / "run.sh")]).returncode == 0
        assert os.access(config.BASE_DIR / "run.sh", os.X_OK), "run.sh должен быть исполняемым"

    def test_окно_с_шаром_на_месте(self):
        from ui import ORB_HTML
        page = ORB_HTML.read_text(encoding="utf-8")
        assert "/state" in page and "pywebview" in page

    def test_все_зависимости_перечислены(self):
        нужны = set((config.BASE_DIR / "requirements.txt").read_text().split())
        for пакет in ["mlx-audio", "mlx-whisper", "vosk", "silero-vad", "sounddevice", "pylibrb",
                      "google-genai", "piper-tts", "pywebview"]:
            assert пакет in нужны, f"{пакет} не записан в requirements.txt"


class TestНастройки:
    def test_ключ_gemini_есть(self):
        assert len(config.GEMINI_API_KEY) > 20, "нет ключа Gemini — впишите его в .env"

    def test_модели_скачаны(self):
        assert config.VOSK_MODEL_PATH.exists(), "нет модели Vosk — запустите ./run.sh"
        assert (config.BASE_DIR / "models" / "piper" / f"{config.PIPER_VOICE}.onnx").exists()

    def test_не_ходим_в_сеть_за_моделями_при_каждом_запуске(self):
        if config._models_downloaded(config.WHISPER_MODEL, config.QWEN_MODEL):
            assert os.environ.get("HF_HUB_OFFLINE") == "1", "проверка обновлений HuggingFace тормозит запуск"

    def test_голос_и_микрофон_выбираются(self):
        import ears
        assert config.TTS_ENGINE in ("qwen", "piper", "gemini", "edge", "silero", "say")
        assert config.QWEN_VOICE_NAME in config.QWEN_VOICES
        ears.pick_microphone()  # не должно падать, даже если микрофона нет

    def test_ctrl_z_закрывает_а_не_замораживает(self):
        import main
        import pytest
        with pytest.raises(KeyboardInterrupt):
            main.exit_on_ctrl_z()
        assert "SIGTSTP" in (config.BASE_DIR / "main.py").read_text()


class TestТекстовыйРежим:
    def test_запускается_и_закрывается(self):
        """Самая грубая проверка: Добрыня стартует, печатает приглашение и выходит."""
        p = subprocess.run([sys.executable, str(config.BASE_DIR / "main.py"), "--text"],
                           input="", capture_output=True, text=True, timeout=180,
                           cwd=str(config.BASE_DIR))
        assert p.returncode == 0, p.stderr[-800:]
        assert config.NAME in p.stdout
