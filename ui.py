"""Шар на экране: состояние Добрыни для окна с анимацией.

Окно работает в отдельном процессе (ui_window.py) и 20 раз в секунду
спрашивает у этого локального сервера, что сейчас происходит.
"""
import atexit
import json
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

import config

ORB_HTML = config.BASE_DIR / "orb.html"
_lock = threading.Lock()
_state = {"state": "loading", "user": "", "answer": ""}
_mic_level = 0.0
_speech = None  # (время начала, огибающая громкости, шаг в секундах)
ENVELOPE_HOP = 1 / 60


def set_state(state: str, user: str | None = None, answer: str | None = None):
    with _lock:
        _state["state"] = state
        if user is not None:
            _state["user"] = user
        if answer is not None:
            _state["answer"] = answer


def mic_level(rms: float):
    global _mic_level
    _mic_level = min(1.0, rms / 3000)


def speaking(audio: np.ndarray, sample_rate: int):
    """Вызывается перед воспроизведением: запоминаем огибающую, чтобы шар пульсировал в такт."""
    global _speech
    hop = max(1, int(sample_rate * ENVELOPE_HOP))
    n = len(audio) // hop
    env = np.sqrt(np.mean(audio[: n * hop].reshape(n, hop) ** 2, axis=1)) if n else np.zeros(1)
    env = np.clip(env / (np.percentile(env, 95) + 1e-9), 0, 1)
    _speech = (time.time(), env, hop / sample_rate)


def speaking_done():
    global _speech
    _speech = None


def _snapshot() -> dict:
    with _lock:
        snap = dict(_state)
    speech = _speech
    if speech:
        start, env, hop = speech
        i = int((time.time() - start) / hop)
        snap["state"], snap["level"] = "speaking", float(env[i]) if i < len(env) else 0.0
    else:
        snap["level"] = _mic_level if snap["state"] == "listening" else 0.0
    return snap


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/state"):
            body, ctype = json.dumps(_snapshot(), ensure_ascii=False).encode(), "application/json"
        else:
            body, ctype = ORB_HTML.read_bytes(), "text/html; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def start():
    """Поднимает сервер состояния и открывает окно с шаром."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    proc = subprocess.Popen([sys.executable, str(config.BASE_DIR / "ui_window.py"), url],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    atexit.register(proc.terminate)
