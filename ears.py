"""Уши: постоянное прослушивание слова активации и запись фразы."""
import json
import re
import queue
import subprocess
import time

import numpy as np
import sounddevice as sd
import vosk

import config
import ui

BLOCK = 1600  # 100 мс при 16 кГц
VAD_WINDOW = 512  # Silero VAD принимает окна по 32 мс
_VAD = None

# галлюцинации Whisper на тишине и шуме (обучался на видео с субтитрами)
JUNK_PATTERNS = [
    r"продолжение следует\W*",
    r"субтитр\w*[^.!?]*[.!?]?",
    r"редактор субтитров[^.!?]*[.!?]?",
    r"корректор [^.!?]*[.!?]?",
    r"спасибо за просмотр\w*\W*",
    r"подписывайтесь на канал\W*",
    r"ставьте лайки\W*",
    r"dimatorzok\W*",
]


def pick_microphone():
    """Номер микрофона из config.MIC_DEVICE (по части названия) или None — системный по умолчанию."""
    if not config.MIC_DEVICE:
        return None
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0 and config.MIC_DEVICE.lower() in dev["name"].lower():
            return i
    print(f"[уши] микрофон «{config.MIC_DEVICE}» не найден, беру системный по умолчанию")
    return None


class Ears:
    def __init__(self):
        vosk.SetLogLevel(-1)
        self.vosk_model = vosk.Model(str(config.VOSK_MODEL_PATH))
        self.q: queue.Queue = queue.Queue()
        device = pick_microphone()
        print(f"[уши] микрофон: {sd.query_devices(device)['name'] if device is not None else sd.query_devices(kind='input')['name']}")
        self.stream = sd.RawInputStream(samplerate=config.SAMPLE_RATE, blocksize=BLOCK, device=device,
                                        dtype="int16", channels=1,
                                        callback=self._on_audio)
        self.stream.start()
        self.noise_level = 300.0
        self.speech_ended = time.time()
        print("[уши] загружаю Whisper…")
        self.transcribe(np.zeros(config.SAMPLE_RATE, dtype=np.float32))  # прогрев

    def _on_audio(self, data, *_):
        chunk = bytes(data)
        self.q.put(chunk)
        ui.mic_level(self._rms(chunk))  # шар реагирует на голос

    def flush(self):
        """Выбросить звук, накопившийся пока Добрыня говорил сам."""
        while not self.q.empty():
            self.q.get_nowait()

    def wait_for_wake_word(self) -> np.ndarray:
        """Ждёт имя и возвращает звук всей фразы, в которой оно прозвучало."""
        grammar = json.dumps(config.WAKE_WORDS + config.WAKE_DECOYS + ["[unk]"], ensure_ascii=False)
        rec = vosk.KaldiRecognizer(self.vosk_model, config.SAMPLE_RATE, grammar)
        rec.SetWords(True)
        self.flush()
        utterance: list[bytes] = []
        while True:
            data = self.q.get()
            self._track_noise(data)
            utterance.append(data)
            if not rec.AcceptWaveform(data):
                utterance = utterance[-100:]  # не больше 10 секунд
                continue
            words = json.loads(rec.Result()).get("result", [])
            audio, utterance = utterance, []
            if not any(w["word"] in config.WAKE_WORDS and w["conf"] >= 0.85 for w in words):
                continue
            # Vosk режет фразу на коротких паузах — дослушиваем до конца
            head = np.frombuffer(b"".join(audio), dtype=np.int16).astype(np.float32) / 32768.0
            tail = self.record_phrase(already_started=True, silence=0.8)
            return np.concatenate([head, tail])

    def _track_noise(self, data: bytes):
        rms = self._rms(data)
        if rms < self.noise_level * 1.5:
            # тихий отрезок — это фон комнаты, подстраиваемся под него
            self.noise_level = max(50.0, 0.95 * self.noise_level + 0.05 * rms)
        else:
            # громко — скорее всего речь; фон почти не трогаем, иначе после фразы оглохнем
            self.noise_level = min(self.noise_level * 1.002, 1500.0)

    @staticmethod
    def _rms(data: bytes) -> float:
        a = np.frombuffer(data, dtype=np.int16).astype(np.float32)
        return float(np.sqrt(np.mean(a * a))) if a.size else 0.0

    def _vad(self):
        """Нейросетевой детектор речи Silero VAD: отличает речь от тишины, а не громкое от тихого."""
        global _VAD
        if _VAD is None:
            from silero_vad import load_silero_vad
            _VAD = load_silero_vad(onnx=True)
        return _VAD

    def _speech_prob(self, data: bytes, state: dict) -> float:
        """Вероятность речи в блоке (максимум по окнам 32 мс)."""
        import torch
        vad = self._vad()
        state["buf"] = np.concatenate([state["buf"], np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768])
        best = 0.0
        while len(state["buf"]) >= VAD_WINDOW:
            window, state["buf"] = state["buf"][:VAD_WINDOW], state["buf"][VAD_WINDOW:]
            best = max(best, vad(torch.from_numpy(window), config.SAMPLE_RATE).item())
        return best

    def record_phrase(self, already_started=False, silence=None, wait=None) -> np.ndarray | None:
        """Пишет звук, пока человек говорит; возвращает None, если так и не заговорил.
        Конец фразы — когда VAD не слышит речь `silence` секунд подряд."""
        silence = silence or config.SILENCE_SECONDS
        wait = wait or config.WAIT_SPEECH_SECONDS
        block_sec = BLOCK / config.SAMPLE_RATE
        self._vad().reset_states()
        vad_state = {"buf": np.zeros(0, dtype=np.float32)}
        chunks, pre = [], []
        started, speech_sec, silent, last_speech = already_started, 0.0, 0.0, 0
        start = time.time()
        while True:
            data = self.q.get()
            is_speech = self._speech_prob(data, vad_state) > 0.5
            if not started:
                pre = (pre + [data])[-5:]  # полсекунды до начала речи, чтобы не срезать первый слог
                if is_speech:
                    started, chunks, speech_sec, silent = True, pre[:], block_sec, 0.0
                    last_speech = len(chunks)
                elif time.time() - start > wait:
                    return None
                continue
            chunks.append(data)
            if is_speech:
                speech_sec, silent, last_speech = speech_sec + block_sec, 0.0, len(chunks)
            else:
                silent += block_sec
            if silent >= silence:
                if already_started or speech_sec >= 0.3:
                    break
                # щелчок, стук или вздох — не речь, ждём дальше
                started, chunks, pre = False, [], []
            if time.time() - start > config.MAX_RECORD_SECONDS:
                break
        self.speech_ended = time.time()
        # хвост тишины Whisper не нужен: на нём он и выдумывает «Продолжение следует»
        chunks = chunks[:last_speech + 3]
        pcm = np.frombuffer(b"".join(chunks), dtype=np.int16)
        return pcm.astype(np.float32) / 32768.0

    def transcribe(self, audio: np.ndarray) -> str:
        import mlx_whisper
        result = mlx_whisper.transcribe(audio, path_or_hf_repo=config.WHISPER_MODEL,
                                        language="ru", condition_on_previous_text=False,
                                        initial_prompt=f"{config.NAME}, ")  # подсказка, как пишется имя
        return result["text"].strip()

    def listen(self, wait=None) -> str:
        audio = self.record_phrase(wait=wait)
        if audio is None:
            return ""
        return self.clean_text(self.transcribe(audio))

    @staticmethod
    def clean_text(text: str) -> str:
        # Whisper иногда дописывает «титры» на тишине — вырезаем их, остальное оставляем
        for pattern in JUNK_PATTERNS:
            text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
        # убираем обращение: «Эй, Добрыня, Добрыня, какая погода?» -> «какая погода?»
        names = "|".join(map(re.escape, config.WAKE_WORDS))
        text = re.sub(rf"^\W*(?:(?:эй|алло?|алё|слушай)\W+)?(?:(?:{names})\W*)+", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text).strip(" ,.!?-—")
        return text if sum(c.isalpha() for c in text) >= 2 else ""


def beep(kind: str = "start"):
    sound = {"start": "Tink", "end": "Pop"}[kind]
    subprocess.Popen(["afplay", f"/System/Library/Sounds/{sound}.aiff"])
