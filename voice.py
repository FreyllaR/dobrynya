"""Голос: Piper (быстрый, локальный), Edge Dmitry (облако), Silero (офлайн) или встроенный say."""
import asyncio
import re
import subprocess
import threading
import time
from xml.sax.saxutils import escape

import numpy as np

import config
import ui

_speak_lock = threading.Lock()
SAMPLE_RATE = 48000
EDGE_DEADLINE = 6  # секунд; дольше — говорим запасным голосом


def _numbers_to_words(text: str) -> str:
    from num2words import num2words

    def repl(m):
        s = m.group(0).replace(",", ".")
        try:
            return num2words(float(s) if "." in s else int(s), lang="ru")
        except Exception:
            return s

    text = re.sub(r"(\d{1,2}):(\d{2})", r"\1 \2", text)  # 18:30 -> 18 30
    text = text.replace("°", " градусов").replace("%", " процентов")
    return re.sub(r"\d+(?:[.,]\d+)?", repl, text)


def apply_stress_fixes(text: str) -> str:
    """Ручные ударения из config.STRESS_FIXES: «+» ставится перед ударной гласной."""
    for word, stressed in config.STRESS_FIXES.items():
        pattern = re.compile(rf"(?<![\w+]){re.escape(word)}(?!\w)", re.IGNORECASE)

        def repl(m, stressed=stressed):
            return stressed[0].upper() + stressed[1:] if m.group(0)[0].isupper() else stressed

        text = pattern.sub(repl, text)
    return text


def plus_to_acute(text: str) -> str:
    """«добр+ыня» -> «добры́ня»: нейросетевые голоса понимают знак ударения, а не плюс."""
    return re.sub(r"\+([аеёиоуыэюяАЕЁИОУЫЭЮЯ])", "\\1\u0301", text)


def clean(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[*_#`>\[\]]", "", text)
    text = re.sub(r"[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F]", "", text)  # эмодзи
    return re.sub(r"\s+", " ", text).strip()


def soft_effect(x: np.ndarray, sr: int = SAMPLE_RATE, softness: float | None = None) -> np.ndarray:
    """Мягкий тембр: без резких верхов и шипящих, теплее, с тихим «воздухом» зала.
    softness от 0 (почти без обработки) до 1 (бархатно, как по радио ночью)."""
    from scipy.signal import butter, sosfilt

    def sos(*args):
        return butter(*args, output="sos")

    soft = config.VOICE_SOFTNESS if softness is None else softness
    x = x.astype(np.float64)
    x = sosfilt(sos(2, 80 / (sr / 2), "high"), x)
    x = sosfilt(sos(2, (8000 - 3500 * soft) / (sr / 2), "low"), x)       # 8 → 4,5 кГц

    # де-эссер: приглушаем «с», «ш», «щ», только когда они громкие
    band = sosfilt(sos(2, [3800 / (sr / 2), 9000 / (sr / 2)], "band"), x)
    env = sosfilt(sos(1, 30 / (sr / 2), "low"), np.abs(band))
    thr = np.percentile(env, 75 - 25 * soft)
    x = x - band * (1 - np.where(env > thr, thr / (env + 1e-9), 1.0) ** (0.8 + 0.4 * soft))

    # тепло в низах и «прикрытая» середина 2–3,5 кГц, где голос звучит жёстко
    x = x + (0.3 + 0.4 * soft) * sosfilt(sos(2, [110 / (sr / 2), 320 / (sr / 2)], "band"), x)
    x = x - (0.35 * soft) * sosfilt(sos(2, [2000 / (sr / 2), 3500 / (sr / 2)], "band"), x)

    # плавная атака: сглаживаем резкие всплески громкости (без «пампинга» компрессора)
    level = sosfilt(sos(1, 8 / (sr / 2), "low"), np.abs(x))
    ceiling = np.percentile(level, 95)
    x = x * np.minimum(1.0, (ceiling / (level + 1e-9)) ** (0.5 * soft))

    x = np.concatenate([x, np.zeros(int(0.5 * sr))])
    reverb = np.zeros_like(x)
    for ms, gain in ((29, .05), (41, .045), (59, .04), (83, .035), (113, .03),
                     (151, .022), (199, .015), (263, .01)):
        k = int(ms / 1000 * sr)
        reverb[k:] += gain * x[:-k]
    x = x + sosfilt(sos(2, 3000 / (sr / 2), "low"), reverb)
    return (config.VOICE_VOLUME * x / (np.abs(x).max() + 1e-9)).astype(np.float32)


class Voice:
    def __init__(self):
        from concurrent.futures import ThreadPoolExecutor
        self.engine = config.TTS_ENGINE
        self.silero = None
        self.net_pool = ThreadPoolExecutor(max_workers=8)
        self.piper = None
        if self.engine == "piper":
            try:
                from piper import PiperVoice
                self.piper = PiperVoice.load(config.BASE_DIR / "models" / "piper" / f"{config.PIPER_VOICE}.onnx")
                self._piper("Прогрев.")
                return
            except Exception as e:
                print(f"[голос] Piper не загрузился ({e}), использую Silero")
                self.engine = "silero"
        if self.engine in ("edge", "silero"):
            # Silero нужен и как запасной голос, если у edge пропадёт интернет
            try:
                import torch
                torch.set_num_threads(4)
                self.silero, _ = torch.hub.load("snakers4/silero-models", "silero_tts", language="ru",
                                                speaker=config.SILERO_MODEL, trust_repo=True,
                                                verbose=False)
            except Exception as e:
                print(f"[голос] Silero не загрузился ({e})")
                if self.engine == "silero":
                    self.engine = "say"

    def say(self, text: str):
        self.say_stream([text])

    def say_stream(self, pieces, on_start=None) -> str:
        """Озвучивает текст, приходящий кусками: каждое готовое предложение сразу
        уходит в синтез, а звучат они строго по порядку. Возвращает весь текст."""
        import queue
        from concurrent.futures import ThreadPoolExecutor

        import sounddevice as sd

        with _speak_lock:
            ready: queue.Queue = queue.Queue()

            def player():
                first = True
                while (job := ready.get()) is not None:
                    audio = job.result() if hasattr(job, "result") else job
                    if first and on_start:
                        on_start()
                    first = False
                    if self.engine == "say":
                        subprocess.run(["say", "-v", config.SAY_VOICE, audio])
                    else:
                        ui.speaking(audio, SAMPLE_RATE)  # шар пульсирует в такт речи
                        sd.play(audio, SAMPLE_RATE)
                        sd.wait()
                ui.speaking_done()

            def submit(sentence):
                sentence = clean(sentence)
                if not re.search(r"[а-яА-Яa-zA-Z]", sentence):
                    return
                if self.engine == "say":
                    ready.put(sentence)
                else:
                    spoken = apply_stress_fixes(_numbers_to_words(sentence))
                    ready.put(pool.submit(self.render, spoken))

            full, buffer = "", ""
            with ThreadPoolExecutor(max_workers=4) as pool:
                play_thread = threading.Thread(target=player, daemon=True)
                play_thread.start()
                for piece in pieces:
                    full += piece
                    buffer += piece
                    # отправляем в синтез все законченные предложения
                    while m := re.search(r"[.!?…]+[\s»\"]+", buffer):
                        submit(buffer[:m.end()])
                        buffer = buffer[m.end():]
                submit(buffer)
                ready.put(None)
                play_thread.join()
            return full.strip()

    def render(self, text: str) -> np.ndarray:
        audio = None
        if self.engine == "piper":
            audio = self._piper(text)
        elif self.engine == "edge":
            try:
                audio = self._edge_fast(text)
            except Exception as e:
                print(f"[голос] edge недоступен ({type(e).__name__}), говорю через Silero")
        if audio is None:
            audio = self._silero(text)
        return soft_effect(audio) if config.VOICE_EFFECT == "soft" else audio

    def _piper(self, text: str) -> np.ndarray:
        """Локальный синтез за сотые доли секунды. Ударения ставит сам, ручные правки — через знак ́."""
        from fractions import Fraction

        from piper import SynthesisConfig
        from scipy.signal import resample_poly
        pitch = config.PIPER_PITCH
        # синтезируем чуть быстрее, потом растягиваем: тон и тембр ниже, темп прежний
        cfg = SynthesisConfig(length_scale=config.PIPER_SPEED * pitch,
                              noise_scale=config.PIPER_NOISE, noise_w_scale=config.PIPER_NOISE_W)
        chunks = [c.audio_float_array for c in self.piper.synthesize(plus_to_acute(text), cfg)]
        audio = np.concatenate(chunks) if chunks else np.zeros(1, dtype=np.float32)
        ratio = Fraction(SAMPLE_RATE / pitch / self.piper.config.sample_rate).limit_denominator(200)
        return resample_poly(audio, ratio.numerator, ratio.denominator).astype(np.float32)

    def _edge_fast(self, text: str) -> np.ndarray:
        """Сервер Edge отвечает то за 0,5 с, то за 5 с. Если первый запрос задержался,
        шлём дубль и берём тот ответ, что придёт раньше: медиана падает с 3,5 до 0,7 с."""
        from concurrent.futures import FIRST_COMPLETED, wait

        start = time.time()
        hedges = [0.8, 1.8]                      # когда отправлять дубли
        pending = {self.net_pool.submit(self._edge, text)}
        while pending and time.time() - start < EDGE_DEADLINE:
            next_hedge = hedges[0] if hedges else EDGE_DEADLINE
            done, pending = wait(pending, timeout=max(0.0, next_hedge - (time.time() - start)),
                                 return_when=FIRST_COMPLETED)
            for f in done:
                if f.exception() is None:
                    return f.result()
            if hedges and time.time() - start >= hedges[0]:
                hedges.pop(0)
                pending.add(self.net_pool.submit(self._edge, text))
            elif not pending and hedges:     # все упали — сразу пробуем снова
                hedges.pop(0)
                pending.add(self.net_pool.submit(self._edge, text))
        raise TimeoutError(f"edge не ответил за {EDGE_DEADLINE} с")

    def _edge(self, text: str) -> np.ndarray:
        import edge_tts

        async def fetch():
            mp3 = b""
            communicate = edge_tts.Communicate(plus_to_acute(text), config.EDGE_VOICE,
                                               rate=config.EDGE_RATE, pitch=config.EDGE_PITCH)
            async for part in communicate.stream():
                if part["type"] == "audio":
                    mp3 += part["data"]
            return mp3

        mp3 = asyncio.run(fetch())
        pcm = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", "pipe:0", "-f", "s16le",
                              "-ac", "1", "-ar", str(SAMPLE_RATE), "pipe:1"],
                             input=mp3, capture_output=True, check=True).stdout
        return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768

    def _silero(self, text: str) -> np.ndarray:
        """Офлайн-синтез. Ударения и «ё» Silero расставляет сам."""
        ssml = (f'<speak><prosody pitch="{config.SILERO_PITCH}" rate="{config.SILERO_RATE}">'
                f"{escape(text[:900])}</prosody></speak>")
        return self.silero.apply_tts(ssml_text=ssml, speaker=config.SILERO_SPEAKER,
                                     sample_rate=SAMPLE_RATE).numpy()
