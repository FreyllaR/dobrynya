"""Окно с шаром: обычное окно macOS по центру экрана. Запускается из ui.py."""
import sys

import webview

WIDTH, HEIGHT = 440, 580


class Api:
    def quit(self):
        for window in webview.windows:
            window.destroy()


def main():
    url = sys.argv[1]
    on_top = len(sys.argv) > 2 and sys.argv[2] == "on-top"
    screen = webview.screens[0]
    webview.create_window(
        "Добрыня", url, js_api=Api(),
        width=WIDTH, height=HEIGHT, min_size=(360, 480),
        x=(screen.width - WIDTH) // 2, y=(screen.height - HEIGHT) // 2,  # по центру экрана
        on_top=on_top, background_color="#FFFFFF",
    )
    webview.start()


if __name__ == "__main__":
    main()
