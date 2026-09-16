"""Мозг: Gemini API (по умолчанию) или локальная LLM через Ollama, с вызовом инструментов."""
import functools
import time

import config
import tools

MAX_TOOL_ROUNDS = 5


def system_prompt() -> str:
    facts = tools.get_memory()
    memory = "\n".join(f"- {f}" for f in facts) if facts else "пока ничего"
    user = f"Хозяина зовут {config.USER_NAME}. " if config.USER_NAME else ""
    return f"""Ты — {config.NAME}, персональный ИИ-ассистент на компьютере Mac, в духе Джарвиса из «Железного человека». {user}

Характер:
- Безупречно вежлив, собран и невозмутим — как лучший британский дворецкий, ставший искусственным интеллектом.
- Обращаешься к пользователю только на «вы» и называешь его «сэр» — примерно раз в одной-двух репликах, естественно, не в каждом предложении.
- Тонкая сухая ирония: можешь деликатно подколоть, если сэр затевает что-то сомнительное, но всегда уважительно и тут же помогаешь.
- Спокойная уверенность: не суетишься, не восторгаешься, не извиняешься лишний раз. Никаких «Отличный вопрос!» и восклицательных знаков через слово.
- Действуешь на опережение: если уместно, коротко предлагаешь следующий шаг («Напомнить об этом вечером, сэр?»).
- Докладываешь чётко: сначала суть, потом деталь. «Готово, сэр» вместо длинных подтверждений.

Правила:
- Отвечай по-русски, коротко: 1–3 предложения. Тебя слушают, а не читают.
- Никакого markdown, списков, эмодзи и ссылок — только живая речь.
- Всегда пиши букву «ё» там, где она нужна (всё, ещё, её) — так тебя правильно озвучат.
- Для действий, актуальных данных, времени, погоды и новостей вызывай инструменты. Не выдумывай факты.
- Если после поиска данных много — перескажи главное своими словами.
- Вызывай remember_fact, только если пользователь прямо просит запомнить. Речь распознаётся с ошибками — не запоминай странные имена и факты без явной просьбы.
- Если запрос непонятен — переспроси коротко.

Что ты помнишь о пользователе:
{memory}"""


def Brain():
    return GeminiBrain() if config.LLM_PROVIDER == "gemini" else OllamaBrain()


class GeminiBrain:
    def __init__(self):
        from google import genai
        from google.genai import types
        if not config.GEMINI_API_KEY:
            raise SystemExit("Нет ключа Gemini. Впиши GEMINI_API_KEY в файл .env (см. README).")
        # без встроенных повторов SDK: при перегрузке он молча ждал до минуты
        self.client = genai.Client(api_key=config.GEMINI_API_KEY, http_options=types.HttpOptions(
            timeout=config.GEMINI_TIMEOUT * 1000, retry_options=types.HttpRetryOptions(attempts=1)))
        self.cooldown: dict[str, float] = {}  # модель -> до какого времени её не трогать
        self.history: list = []
        self.on_tool = None

    def reset(self):
        self.history = []

    def _tools(self):
        def wrap(fn):
            @functools.wraps(fn)
            def logged(*args, **kwargs):
                if self.on_tool:
                    self.on_tool(fn.__name__, kwargs)
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    return f"Ошибка: {e}"
            return logged
        return [wrap(f) for f in tools.TOOLS]

    def _trim(self, history):
        # обрезаем только по границе реплики пользователя, чтобы не разорвать вызов инструмента
        starts = [i for i, c in enumerate(history)
                  if c.role == "user" and any(p.text for p in (c.parts or []))]
        while len(history) > config.HISTORY_LIMIT and len(starts) > 1:
            history = history[starts[1]:]
            starts = [i - starts[1] for i in starts[1:]]
        return history

    @staticmethod
    def _compact(history):
        """Стрим сохраняет ответ кусками — склеиваем соседние текстовые куски модели."""
        from google.genai import types
        out = []
        for c in history:
            only_text = all(p.text is not None and not p.function_call for p in (c.parts or []))
            prev = out[-1] if out else None
            if (prev is not None and c.role == "model" and prev.role == "model" and only_text
                    and all(p.text is not None for p in prev.parts or [])):
                text = "".join(p.text for p in prev.parts) + "".join(p.text for p in c.parts)
                out[-1] = types.Content(role="model", parts=[types.Part(text=text)])
            else:
                out.append(c)
        return out

    def ask(self, text: str, on_tool=None) -> str:
        return "".join(self.ask_stream(text, on_tool)).strip()

    def ask_stream(self, text: str, on_tool=None):
        """Отдаёт ответ кусками по мере генерации — голос начинает говорить раньше."""
        from google.genai import errors, types
        self.on_tool = on_tool
        tools_list = self._tools()
        models = [m for m in config.GEMINI_MODELS if self.cooldown.get(m, 0) < time.time()]
        last_error = None
        # если модель перегружена или кончился бесплатный лимит — пробуем следующую
        for model in models or config.GEMINI_MODELS:
            cfg = types.GenerateContentConfig(
                system_instruction=system_prompt(),
                tools=tools_list,
                # для голоса скорость важнее долгих размышлений
                thinking_config=types.ThinkingConfig(
                    thinking_level="minimal" if "lite" in model else "low"),
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    maximum_remote_calls=MAX_TOOL_ROUNDS * 2),
            )
            chat = self.client.chats.create(model=model, config=cfg, history=self.history)
            started = False
            try:
                for chunk in chat.send_message_stream(text):
                    content = chunk.candidates[0].content if chunk.candidates else None
                    piece = "".join(p.text for p in (content.parts or []) if p.text and not p.thought) \
                        if content else ""
                    if piece:
                        started = True
                        yield piece
            except Exception as e:  # APIError или таймаут сети
                if started:  # оборвалось на середине — модель уже не сменить
                    yield " Связь оборвалась."
                    return
                last_error = e
                code = getattr(e, "code", None)
                if isinstance(e, errors.APIError) and code not in (404, 408, 429, 500, 503, 504):
                    raise
                print(f"[мозг] {model}: {code or type(e).__name__}, пробую другую модель")
                self.cooldown[model] = time.time() + (3600 if code == 404 else 300)
                continue
            self.history = self._trim(self._compact(chat.get_history()))
            if not started:
                yield "Готово."
            return
        if "location" in str(last_error).lower():
            yield "Гугл не пускает из твоей страны. Нужен VPN."
        else:
            yield "Мозги перегрелись, лимиты кончились. Попробуй чуть позже."


class OllamaBrain:
    def __init__(self):
        self.history: list = []

    def ask_stream(self, text: str, on_tool=None):
        yield self.ask(text, on_tool)

    def reset(self):
        self.history = []

    def ask(self, text: str, on_tool=None) -> str:
        import ollama
        self.history.append({"role": "user", "content": text})
        self.history = self.history[-config.HISTORY_LIMIT:]
        # история не должна начинаться с ответа инструмента
        while self.history and self.history[0]["role"] != "user":
            self.history.pop(0)

        for _ in range(MAX_TOOL_ROUNDS):
            messages = [{"role": "system", "content": system_prompt()}] + self.history
            resp = ollama.chat(model=config.OLLAMA_MODEL, messages=messages,
                               tools=tools.TOOLS, think=False, keep_alive="30m")
            msg = resp.message
            self.history.append({"role": "assistant", "content": msg.content or "",
                                 "tool_calls": msg.tool_calls or []})
            if not msg.tool_calls:
                return (msg.content or "").strip()

            for call in msg.tool_calls:
                name, args = call.function.name, dict(call.function.arguments or {})
                if on_tool:
                    on_tool(name, args)
                fn = tools.TOOL_MAP.get(name)
                try:
                    result = fn(**args) if fn else f"Нет инструмента {name}"
                except Exception as e:  # инструмент упал — пусть модель объяснит
                    result = f"Ошибка: {e}"
                self.history.append({"role": "tool", "content": str(result), "tool_name": name})

        return "Что-то я запутался. Попробуй спросить иначе."
