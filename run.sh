#!/bin/bash
# Запуск Добрыни. При первом запуске ставит всё нужное.
set -e
cd "$(dirname "$0")"

if grep -q '^LLM_PROVIDER = "ollama"' config.py; then
  command -v ollama >/dev/null || brew install ollama
  pgrep -x ollama >/dev/null || brew services start ollama
  sleep 1
  ollama list | grep -q "qwen3:8b" || ollama pull qwen3:8b
elif ! grep -q "GEMINI_API_KEY=." .env 2>/dev/null && [ -z "$GEMINI_API_KEY" ]; then
  echo "Впиши ключ Gemini в файл .env (получить бесплатно: https://aistudio.google.com/apikey)"
  exit 1
fi

if [ ! -d .venv ]; then
  python3 -m venv .venv
  .venv/bin/pip install -r requirements.txt
fi

if [ ! -d models/vosk-model-small-ru-0.22 ]; then
  mkdir -p models && cd models
  curl -LO https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip
  unzip -q vosk-model-small-ru-0.22.zip && rm vosk-model-small-ru-0.22.zip
  cd ..
fi

for v in dmitri denis ruslan; do
  for ext in onnx onnx.json; do
    f="models/piper/ru_RU-$v-medium.$ext"
    [ -f "$f" ] || { mkdir -p models/piper; curl -sL -o "$f" "https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/$v/medium/ru_RU-$v-medium.$ext"; }
  done
done

exec .venv/bin/python main.py "$@"
