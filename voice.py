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
GEMINI_SR = 24000  # Gemini TTS отдаёт 16-битный PCM 24 кГц
QWEN_PREBUFFER = 0.5  # секунд звука копим перед стартом, чтобы не было рывков
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


TAG_RE = re.compile(r"\[[a-zA-Z ,'-]{2,40}\]\s*")  # звуковые теги: [sighs], [laughs softly]


def strip_tags(text: str) -> str:
    """Убирает звуковые теги — для субтитров и голосов, которые их не понимают."""
    return re.sub(r"\s+", " ", TAG_RE.sub("", text)).strip()


def qwen_text(text: str) -> str:
    """Для Qwen3-TTS: звуковые теги превращаем в живые междометия, которые он произносит естественно."""
    text = re.sub(r"\[sighs?\]", "Эх...", text, flags=re.IGNORECASE)
    text = re.sub(r"\[(laughs?|giggles?|chuckles?)[^\]]*\]", "Ха-ха,", text, flags=re.IGNORECASE)
    text = clean(text).replace("+", "")
    for word, spoken in config.NAME_PRONUNCIATION.items():  # написание «для слуха», если модель путает ударение
        text = re.sub(rf"\b{re.escape(word)}\b", spoken, text)
    return text


def clean(text: str, keep_tags: bool = False) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"https?://\S+", "", text)
    if not keep_tags:
        text = TAG_RE.sub("", text)
    text = re.sub(r"[*_#`>]", "", text)
    if not keep_tags:
        text = re.sub(r"[\[\]]", "", text)
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
        self.gemini = None
        self.gemini_cooldown = 0.0
        if self.engine == "gemini":
            from google import genai
            from google.genai import types
            self.gemini = genai.Client(api_key=config.GEMINI_API_KEY, http_options=types.HttpOptions(
                timeout=60_000, retry_options=types.HttpRetryOptions(attempts=1)))
        self.qwen = None
        if self.engine == "qwen":
            try:
                from mlx_audio.tts.utils import load_model
                print("[голос] загружаю Qwen3-TTS…")
                self.qwen = load_model(config.QWEN_MODEL)
                for _ in self._qwen_chunks("Готов."):  # прогрев: первая генерация компилирует модель
                    pass
            except Exception as e:
                print(f"[голос] Qwen3-TTS не загрузился ({e}), использую Piper")
        if self.engine in ("piper", "gemini", "qwen"):  # для облачных и Qwen Piper — запасной голос
            try:
                from piper import PiperVoice
                self.piper = PiperVoice.load(config.BASE_DIR / "models" / "piper" / f"{config.PIPER_VOICE}.onnx")
                self._piper("Прогрев.")
                return
            except Exception as e:
                print(f"[голос] Piper не загрузился ({e}), использую Silero")
                if self.engine == "piper":
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
            if self.engine == "qwen" and self.qwen:
                return self._qwen_say_stream(pieces, on_start)
            if self.engine == "gemini":
                # ответ целиком: одна интонация на всю реплику и один запрос к лимиту
                text = "".join(pieces)
                if time.time() >= self.gemini_cooldown:
                    try:
                        self._gemini_say(text, on_start)
                        return strip_tags(text)
                    except Exception as e:
                        code = getattr(e, "code", None)
                        pause = 600 if code == 429 else 120
                        self.gemini_cooldown = time.time() + pause
                        print(f"[голос] Gemini TTS: {code or type(e).__name__} — {pause // 60} мин говорю через Piper")
                        on_start, pieces = on_start, [text]
                else:
                    pieces = [text]

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
        if self.engine in ("piper", "gemini", "qwen") and self.piper:
            audio = self._piper(text)
        elif self.engine == "edge":
            try:
                audio = self._edge_fast(text)
            except Exception as e:
                print(f"[голос] edge недоступен ({type(e).__name__}), говорю через Silero")
        if audio is None:
            audio = self._silero(text)
        return soft_effect(audio) if config.VOICE_EFFECT == "soft" else audio

    def _qwen_chunks(self, text: str):
        """Потоковый синтез голосом богатыря (копия образца voices/dobrynya.wav)."""
        import pylibrb

        # модель говорит медленно, а скорость не настраивается — ускоряем готовый звук без изменения тона.
        # Rubber Band в потоковом режиме: в отличие от WSOLA не даёт «шлепков» на стыках кусков
        rb = None
        if config.QWEN_SPEED != 1.0:
            O = pylibrb.Option
            rb = pylibrb.RubberBandStretcher(
                self.qwen.sample_rate, 1,
                O.PROCESS_REALTIME | O.ENGINE_FINER | O.FORMANT_PRESERVED | O.WINDOW_STANDARD
                | O.TRANSIENTS_SMOOTH | O.PHASE_LAMINAR | O.PitchHighConsistency,
                initial_time_ratio=1 / config.QWEN_SPEED)
            rb.process(np.zeros((1, rb.get_preferred_start_pad()), dtype=np.float32))
            skip = rb.get_start_delay()  # первые отсчёты — задержка алгоритма, их выбрасываем

        def drain(final=False):
            nonlocal skip
            while rb.available() > 0:
                out = rb.retrieve_available()[0]
                if skip:
                    cut = min(skip, len(out)); out = out[cut:]; skip -= cut
                if len(out):
                    yield out
            if final:
                return

        for result in self.qwen.generate(text=text, ref_audio=str(config.QWEN_VOICE), ref_text=config.QWEN_VOICE_TEXT,
                                         lang_code="russian", temperature=config.QWEN_TEMPERATURE,
                                         stream=True, streaming_interval=0.4):
            audio = np.array(result.audio, dtype=np.float32)
            if rb is None:
                yield audio
                continue
            rb.process(np.ascontiguousarray(audio[None, :]))
            yield from drain()
        if rb is not None:
            rb.process(np.zeros((1, 0), dtype=np.float32), final=True)
            yield from drain(final=True)

    def _qwen_say_stream(self, pieces, on_start=None) -> str:
        """Каждое готовое предложение сразу уходит в синтез. Генерация и воспроизведение —
        в разных потоках: модель наполняет очередь с опережением, колонки играют из неё без пауз."""
        import queue

        import sounddevice as sd
        from scipy.signal import resample_poly

        sr = self.qwen.sample_rate
        sentences: queue.Queue = queue.Queue()
        audio_q: queue.Queue = queue.Queue()

        def generator():
            while (sentence := sentences.get()) is not None:
                try:
                    for audio in self._qwen_chunks(sentence):
                        audio_q.put(audio * config.QWEN_VOLUME)
                except Exception as e:  # сбой модели — это предложение скажет Piper
                    print(f"[голос] Qwen3-TTS: {e} — говорю через Piper")
                    if self.piper:
                        audio_q.put(resample_poly(self._piper(strip_tags(sentence)), sr, SAMPLE_RATE).astype(np.float32))
            audio_q.put(None)

        def player():
            # запас перед стартом: колонки не догонят модель на первых словах
            pending, buffered, done = [], 0.0, False
            while buffered < QWEN_PREBUFFER and not done:
                item = audio_q.get()
                if item is None:
                    done = True
                else:
                    pending.append(item)
                    buffered += len(item) / sr
            if not pending:
                return
            if on_start:
                on_start()
            with sd.OutputStream(samplerate=sr, channels=1, dtype="float32") as out:
                while True:
                    for audio in pending:
                        ui.speaking(audio, sr)  # шар двигается в такт речи
                        out.write(audio.reshape(-1, 1))
                    if done:
                        break
                    item = audio_q.get()
                    if item is None:
                        break
                    pending = [item]
            ui.speaking_done()

        gen_thread = threading.Thread(target=generator, daemon=True)
        play_thread = threading.Thread(target=player, daemon=True)
        gen_thread.start()
        play_thread.start()

        def submit(sentence):
            spoken = qwen_text(sentence)
            if re.search(r"[а-яА-Яa-zA-Z]", spoken):
                sentences.put(spoken)

        full, buffer = "", ""
        for piece in pieces:
            full += piece
            buffer += piece
            while m := re.search(r"[.!?…]+[\s»\"]+", buffer):
                submit(buffer[:m.end()])
                buffer = buffer[m.end():]
        submit(buffer)
        sentences.put(None)
        play_thread.join()
        return strip_tags(full)

    def _gemini_say(self, text: str, on_start=None):
        """Живой голос Gemini TTS: интонации, вздохи, смешки. Звук играет по мере генерации."""
        import queue

        import sounddevice as sd
        from google.genai import types

        spoken = plus_to_acute(apply_stress_fixes(clean(text, keep_tags=True)))
        if not re.search(r"[а-яА-Яa-zA-Z]", strip_tags(spoken)):
            return
        cfg = types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=config.GEMINI_VOICE))))
        chunks: queue.Queue = queue.Queue()

        def produce():
            try:
                stream = self.gemini.models.generate_content_stream(
                    model=config.GEMINI_TTS_MODEL, contents=f"{config.GEMINI_VOICE_STYLE}\n\n{spoken}", config=cfg)
                for part_chunk in stream:
                    content = part_chunk.candidates[0].content if part_chunk.candidates else None
                    for part in (content.parts or []) if content else []:
                        if part.inline_data and part.inline_data.data:
                            chunks.put(part.inline_data.data)
                chunks.put(None)
            except Exception as e:
                chunks.put(e)

        threading.Thread(target=produce, daemon=True).start()

        # копим первые 0,4 с звука, чтобы не было рывков; ошибка до первого звука — наверх, к запасному голосу
        pcm, deadline = b"", time.time() + config.GEMINI_TTS_TIMEOUT
        done = False
        while len(pcm) < GEMINI_SR * 2 * 0.4:
            try:
                item = chunks.get(timeout=max(0.05, deadline - time.time()))
            except queue.Empty:
                raise TimeoutError(f"нет звука за {config.GEMINI_TTS_TIMEOUT} с")
            if isinstance(item, Exception):
                raise item
            if item is None:
                done = True
                break
            pcm += item
        if not pcm:
            return

        with sd.OutputStream(samplerate=GEMINI_SR, channels=1, dtype="float32") as out:
            if on_start:
                on_start()
            while True:
                usable = len(pcm) - len(pcm) % 2  # кусок может разрезать 16-битный отсчёт пополам
                if usable:
                    audio = np.frombuffer(pcm[:usable], dtype=np.int16).astype(np.float32) / 32768
                    audio *= config.GEMINI_VOLUME
                    ui.speaking(audio, GEMINI_SR)  # шар пульсирует в такт речи
                    out.write(audio.reshape(-1, 1))
                    pcm = pcm[usable:]
                if done:
                    break
                item = chunks.get(timeout=config.GEMINI_TTS_TIMEOUT)
                if isinstance(item, Exception):
                    print(f"[голос] Gemini TTS оборвался: {item}")
                    break
                if item is None:
                    done = True
                else:
                    pcm += item
        ui.speaking_done()

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
