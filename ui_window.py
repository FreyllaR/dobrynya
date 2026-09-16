"""Окно с шаром: прозрачное, без рамки, поверх остальных окон. Запускается из ui.py."""
import sys

import webview

WIDTH, HEIGHT = 320, 400


class Api:
    def quit(self):
        for window in webview.windows:
            window.destroy()


def main():
    url = sys.argv[1]
    screen = webview.screens[0]
    webview.create_window(
        "Добрыня", url, js_api=Api(),
        width=WIDTH, height=HEIGHT,
        x=screen.width - WIDTH - 24, y=screen.height - HEIGHT - 60,  # правый нижний угол
        frameless=True, easy_drag=True, transparent=True, on_top=True,
        resizable=False, shadow=False, background_color="#000000",
    )
    webview.start()


if __name__ == "__main__":
    main()
