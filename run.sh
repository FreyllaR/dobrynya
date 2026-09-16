#!/bin/bash
# Запуск Добрыни. При первом запуске ставит всё нужное.
set -e
cd "$(dirname "$0")"

if grep -q '^LLM_PROVIDER = "ollama"' config.py; then
  command -v ollama >/dev/null || brew install ollama
  pgrep -x ollama >/dev/null || brew services start ollama
  sleep 1
  ollama list | grep -q "qwen3:8b" || ollama pull qwen3:8b
elif [ -z "$GEMINI_API_KEY" ] && ! LC_ALL=C grep -qE "^GEMINI_API_KEY=[!-~]{20,}" .env 2>/dev/null; then
  echo "Нет ключа Gemini. Получите его бесплатно: https://aistudio.google.com/apikey"
  echo "и впишите в файл .env строкой: GEMINI_API_KEY=ваш_настоящий_ключ"
  exit 1
fi

# Зависимости ставим заново, если окружения нет, прошлая установка прервалась
# или изменился requirements.txt
STAMP=".venv/.requirements.sha"
REQ_SHA=$(shasum requirements.txt | cut -d' ' -f1)
if [ ! -x .venv/bin/python ] || [ "$(cat "$STAMP" 2>/dev/null)" != "$REQ_SHA" ]; then
  [ -x .venv/bin/python ] || python3 -m venv .venv
  # pip 24.2+ проверяет сертификаты через связку ключей macOS, и на некоторых Маках
  # это падает с CERTIFICATE_VERIFY_FAILED. Даём ему встроенный набор сертификатов.
  CERT=$(.venv/bin/python -c "import pip._vendor.certifi as c; print(c.where())")
  .venv/bin/python -m pip install --cert "$CERT" -r requirements.txt
  echo "$REQ_SHA" > "$STAMP"
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
