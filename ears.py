"""Уши: постоянное прослушивание слова активации и запись фразы."""
import json
import queue
import subprocess
import time

import numpy as np
import sounddevice as sd
import vosk

import config
import ui

BLOCK = 1600  # 100 мс при 16 кГц


class Ears:
    def __init__(self):
        vosk.SetLogLevel(-1)
        self.vosk_model = vosk.Model(str(config.VOSK_MODEL_PATH))
        self.q: queue.Queue = queue.Queue()
        self.stream = sd.RawInputStream(samplerate=config.SAMPLE_RATE, blocksize=BLOCK,
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

    def record_phrase(self, already_started=False, silence=None, wait=None) -> np.ndarray | None:
        """Пишет звук, пока человек говорит; возвращает None, если так и не заговорил."""
        threshold = min(max(self.noise_level * 2.5, 400), 2500)
        silence = silence or config.SILENCE_SECONDS
        wait = wait or config.WAIT_SPEECH_SECONDS
        block_sec = BLOCK / config.SAMPLE_RATE
        chunks, pre = [], []
        started, loud_blocks, silent, start = already_started, 0, 0.0, time.time()
        while True:
            data = self.q.get()
            loud = self._rms(data) > threshold
            if not started:
                pre = (pre + [data])[-3:]  # 300 мс до начала речи, чтобы не срезать первый слог
                if loud:
                    started, chunks, loud_blocks, silent = True, pre[:], 1, 0.0
                elif time.time() - start > wait:
                    return None
                continue
            chunks.append(data)
            if loud:
                loud_blocks, silent = loud_blocks + 1, 0.0
            else:
                silent += block_sec
            if silent >= silence:
                if already_started or loud_blocks >= 3:
                    break
                # короткий щелчок или стук — не речь, ждём дальше
                started, chunks, pre = False, [], []
            if time.time() - start > config.MAX_RECORD_SECONDS:
                break
        self.speech_ended = time.time()
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
        # Whisper иногда «слышит» титры в тишине
        junk = ("субтитры", "продолжение следует", "спасибо за просмотр", "редактор субтитров")
        if any(j in text.lower() for j in junk):
            return ""
        # убираем обращение: «Эй, Добрыня, какая погода?» -> «какая погода?»
        low = text.lower()
        for w in config.WAKE_WORDS:
            i = low.find(w)
            if i != -1 and i < 8:
                text = text[i + len(w):]
                break
        text = text.lstrip(" ,.!?-—")
        return text if sum(c.isalpha() for c in text) >= 2 else ""


def beep(kind: str = "start"):
    sound = {"start": "Tink", "end": "Pop"}[kind]
    subprocess.Popen(["afplay", f"/System/Library/Sounds/{sound}.aiff"])
