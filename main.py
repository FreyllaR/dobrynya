"""Добрыня — голосовой помощник.

Запуск:
    ./run.sh          — голосовой режим (скажи «Добрыня»)
    ./run.sh --text   — текстовый режим, удобно для отладки навыков
"""
import signal
import sys
import time

import config
import tools
import ui
from brain import Brain
from voice import strip_tags

STOP_WORDS = {"стоп", "хватит", "отбой", "пока", "всё", "все", "спасибо", "отдыхай", "свободен"}


def is_stop(text: str) -> bool:
    """«Стоп», «спасибо, всё», «пока, Добрыня» — конец разговора. Длинные фразы — не стоп."""
    words = [w.strip(" ,.!?—-") for w in text.lower().split()]
    words = [w for w in words if w and w not in config.WAKE_WORDS]
    return 0 < len(words) <= 3 and all(w in STOP_WORDS for w in words)


def show_tool(name, args):
    print(f"  ⚙ {name}({', '.join(f'{k}={v!r}' for k, v in args.items())})")


def text_mode():
    brain = Brain()
    tools.set_notifier(lambda t: print(f"\n🔔 {config.NAME}: {t}\n> ", end=""))
    print(f"{config.NAME} слушает (текстом). Пустая строка — выход.")
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            break
        print(f"{config.NAME}: ", end="", flush=True)
        for piece in brain.ask_stream(text, on_tool=show_tool):
            print(piece, end="", flush=True)
        print()


def voice_mode():
    from ears import Ears, beep
    from voice import Voice

    print("Загружаюсь…")
    if config.SHOW_ORB:
        ui.start()
    voice = Voice()
    ears = Ears()
    brain = Brain()

    def notify(text):
        print(f"🔔 {text}")
        voice.say(text)
        ears.flush()

    tools.set_notifier(notify)
    last_talk = 0.0

    voice.say("К вашим услугам, сэр.")
    print(f"\nГотов. Скажи «{config.NAME}». Ctrl+C — выход.\n")

    while True:
        ui.set_state("idle", user="", answer="")
        phrase = ears.wait_for_wake_word()
        ui.set_state("listening")
        if time.time() - last_talk > 300:  # 5 минут тишины — начинаем разговор заново
            brain.reset()
        # «Добрыня, какая погода» одной фразой — команда уже в записи
        text = ears.clean_text(ears.transcribe(phrase))
        if not text:
            beep("start")
            print("🎙  слушаю…")
            text = ears.listen()

        # режим диалога: после ответа слушаем дальше без слова «Добрыня»
        while text:
            heard_at = ears.speech_ended
            print(f"Ты: {text}")
            ui.set_state("thinking", user=text, answer="")
            if is_stop(text):
                voice.say("Как скажете, сэр. Буду рядом.")
                break

            timings = {"распознал": time.time() - heard_at}
            asked_at = time.time()

            answer_parts = []

            def pieces():
                first = True
                for piece in brain.ask_stream(text, on_tool=show_tool):
                    if first:
                        timings["первые слова"] = time.time() - asked_at
                        print(f"{config.NAME}: ", end="", flush=True)
                        first = False
                    print(piece, end="", flush=True)
                    answer_parts.append(piece)
                    ui.set_state("thinking", answer=strip_tags("".join(answer_parts)))
                    yield piece
                print()

            def started():
                timings["голос через"] = time.time() - heard_at

            voice.say_stream(pieces(), on_start=started)
            if config.SHOW_TIMINGS:
                print("⏱  " + " · ".join(f"{k} {v:.1f} с" for k, v in timings.items()))

            time.sleep(0.2)  # хвост эха из колонок
            ears.flush()
            last_talk = time.time()
            print(f"🎙  слушаю ({config.CONVERSATION_TIMEOUT} с, «стоп» — закончить)…")
            ui.set_state("listening", user="")
            text = ears.listen(wait=config.CONVERSATION_TIMEOUT)

        beep("end")
        print(f"💤 жду «{config.NAME}»\n")


def exit_on_ctrl_z(*_):
    # Ctrl+Z в терминале не закрывает программу, а замораживает её вместе с моделями и микрофоном.
    # Чтобы в памяти не копились «зависшие» Добрыни, считаем Ctrl+Z выходом.
    raise KeyboardInterrupt


if __name__ == "__main__":
    signal.signal(signal.SIGTSTP, exit_on_ctrl_z)
    try:
        if "--mic-test" in sys.argv:
            import mic_test
            mic_test.main()
        elif "--text" in sys.argv:
            text_mode()
        else:
            voice_mode()
    except KeyboardInterrupt:
        print("\nДо встречи!")
