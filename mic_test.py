"""Проверка микрофона и распознавания: ./run.sh --mic-test

Записывает 15 секунд с микрофона, сохраняет в data/mic_test.wav и показывает,
что на каждом этапе услышал Добрыня: громкость, слово активации, границы фраз, текст.
"""
import json
import time
import wave

import numpy as np
import sounddevice as sd
import vosk

import config
from ears import BLOCK, Ears

SECONDS = 15
PATH = config.DATA_DIR / "mic_test.wav"


def main():
    dev = sd.query_devices(kind="input")
    print(f"Микрофон: {dev['name']}\n")
    print("Запись длится 15 секунд. Скажите обычным голосом, как говорите с Добрыней, с паузами:")
    print("  1) «Добрыня»")
    print("  2) «Добрыня, какая погода в Москве?»")
    print("  3) «Напомни мне завтра в девять утра, э-э, позвонить маме»\n")
    input("Нажмите Enter и начинайте говорить…")
    print("🔴 Запись…")
    audio = sd.rec(SECONDS * config.SAMPLE_RATE, samplerate=config.SAMPLE_RATE, channels=1, dtype="int16")
    for i in range(SECONDS, 0, -1):
        print(f"\r   осталось {i:2d} с", end="", flush=True)
        time.sleep(1)
    sd.wait()
    print("\r⏹  Готово.          \n")
    pcm = audio[:, 0]
    with wave.open(str(PATH), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(config.SAMPLE_RATE); w.writeframes(pcm.tobytes())
    analyze(pcm)


def analyze(pcm: np.ndarray):
    blocks = [pcm[i:i + BLOCK] for i in range(0, len(pcm) - BLOCK + 1, BLOCK)]
    rms = np.array([np.sqrt(np.mean(b.astype(np.float32) ** 2)) for b in blocks])
    noise, speech = np.percentile(rms, 10), np.percentile(rms, 90)
    threshold = min(max(noise * 2.5, 400), 2500)
    print("— Громкость (RMS) —")
    print(f"  фон комнаты: {noise:.0f} · речь: {speech:.0f} · пик: {rms.max():.0f} · порог Добрыни: {threshold:.0f}")
    print(f"  речь громче порога в {speech / threshold:.1f} раза" + ("  ⚠️ СЛИШКОМ ТИХО" if speech < threshold * 1.5 else ""))
    print("  по секундам: " + " ".join(f"{rms[i:i + 10].max():.0f}" for i in range(0, len(rms), 10)))

    print("\n— Слово активации (Vosk) —")
    vosk.SetLogLevel(-1)
    model = vosk.Model(str(config.VOSK_MODEL_PATH))
    grammar = json.dumps(config.WAKE_WORDS + config.WAKE_DECOYS + ["[unk]"], ensure_ascii=False)
    rec = vosk.KaldiRecognizer(model, config.SAMPLE_RATE, grammar)
    rec.SetWords(True)
    found = False
    for b in blocks:
        if rec.AcceptWaveform(b.tobytes()):
            for w in json.loads(rec.Result()).get("result", []):
                mark = "✅" if w["word"] in config.WAKE_WORDS and w["conf"] >= 0.85 else "  "
                found |= mark == "✅"
                print(f"  {mark} {w['start']:5.1f} с  «{w['word']}»  уверенность {w['conf']:.2f}")
    for w in json.loads(rec.FinalResult()).get("result", []):
        print(f"     {w['start']:5.1f} с  «{w['word']}»  уверенность {w['conf']:.2f}")
    if not found:
        print("  ⚠️ «Добрыня» не распознан с уверенностью ≥ 0.85")

    print("\n— Границы фраз по текущему алгоритму —")
    block_sec = BLOCK / config.SAMPLE_RATE
    start, silent = None, 0.0
    for i, r in enumerate(rms):
        if r > threshold:
            start = i * block_sec if start is None else start
            silent = 0.0
        elif start is not None:
            silent += block_sec
            if silent >= config.SILENCE_SECONDS:
                print(f"  фраза {start:4.1f}–{i * block_sec - silent:4.1f} с")
                start = None
    if start is not None:
        print(f"  фраза {start:4.1f}–{SECONDS} с (не закончилась)")

    print("\n— Whisper (весь фрагмент) —")
    ears = Ears.__new__(Ears)
    print("  " + ears.transcribe(pcm.astype(np.float32) / 32768.0))
    print(f"\nЗапись сохранена: {PATH}")


if __name__ == "__main__":
    main()
