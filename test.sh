#!/bin/bash
# Проверка Добрыни: запускайте до и после правок.
#
#   ./test.sh          всё, кроме обращений к Gemini (~2 минуты)
#   ./test.sh quick    только быстрые проверки, без моделей (~5 секунд)
#   ./test.sh net      плюс живой разговор с Gemini (тратит бесплатный лимит)
set -e
cd "$(dirname "$0")"

[ -x .venv/bin/python ] || { echo "Нет окружения — сначала запустите ./run.sh"; exit 1; }
.venv/bin/python -c "import pytest, pytest_timeout" 2>/dev/null || {
  CERT=$(.venv/bin/python -c "import pip._vendor.certifi as c; print(c.where())")
  .venv/bin/python -m pip install --quiet --cert "$CERT" pytest pytest-timeout
}

case "${1:-}" in
  quick) exec .venv/bin/python -m pytest --quick ;;
  net)   exec .venv/bin/python -m pytest --net ;;
  "")    exec .venv/bin/python -m pytest ;;
  *)     exec .venv/bin/python -m pytest "$@" ;;
esac
