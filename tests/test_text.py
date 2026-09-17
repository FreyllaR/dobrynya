"""Обработка текста: что Добрыня слышит и что уходит в голос."""
import config
import main
import voice
from ears import Ears


class TestРаспознанныйТекст:
    def test_обращение_вырезается(self):
        assert Ears.clean_text("Добрыня, какая погода?") == "какая погода"
        assert Ears.clean_text("Эй, Добрыня, Добрыня, открой сафари") == "открой сафари"
        assert Ears.clean_text("Алло, Добрыня, ты здесь?") == "ты здесь"

    def test_только_имя_это_не_команда(self):
        assert Ears.clean_text("Добрыня.") == ""
        assert Ears.clean_text("") == ""

    def test_галлюцинации_whisper_вырезаются_а_фраза_остаётся(self):
        # раньше фильтр выбрасывал всю фразу целиком — Добрыня «глох» после каждой команды
        assert Ears.clean_text("Какая погода в Москве? Продолжение следует...") == "Какая погода в Москве"
        assert Ears.clean_text("Поставь таймер. Спасибо за просмотр!") == "Поставь таймер"
        assert Ears.clean_text("Субтитры сделал DimaTorzok") == ""

    def test_обычные_слова_не_страдают(self):
        assert Ears.clean_text("Добрый вечер, что там по погоде") == "Добрый вечер, что там по погоде"


class TestКонецРазговора:
    def test_короткие_команды_завершают(self):
        for phrase in ["стоп", "Спасибо, всё.", "пока, Добрыня", "хватит"]:
            assert main.is_stop(phrase), phrase

    def test_длинные_фразы_не_завершают(self):
        for phrase in ["всё понятно, а что завтра?", "спасибо, а какая погода", "стоп, а можно погромче"]:
            assert not main.is_stop(phrase), phrase


class TestТекстДляГолоса:
    def test_разметка_и_эмодзи_не_читаются_вслух(self):
        assert voice.clean("**Готово**, сэр 🙂 https://example.com") == "Готово, сэр"

    def test_звуковые_теги_не_попадают_в_субтитры(self):
        assert voice.strip_tags("[sighs] Дождь, сэр. [laughs softly] Шучу.") == "Дождь, сэр. Шучу."

    def test_междометия_выключены(self, monkeypatch):
        monkeypatch.setattr(config, "VOICE_NONVERBAL", False)
        assert voice.qwen_text("Хм... замок закрыт, сэр.") == "Замок закрыт, сэр."
        assert voice.qwen_text("[sighs] Эх, погода не радует.") == "Погода не радует."
        assert voice.qwen_text("Охрана на месте.") == "Охрана на месте."  # не путаем с «ох»

    def test_междометия_включены(self, monkeypatch):
        monkeypatch.setattr(config, "VOICE_NONVERBAL", True)
        assert voice.qwen_text("[sighs] Погода не радует.").startswith("Эх...")

    def test_произношение_имени_подставляется(self, monkeypatch):
        monkeypatch.setattr(config, "NAME_PRONUNCIATION", {"Ольков": "Олькофф"})
        assert voice.qwen_text("Готово, мистер Ольков.") == "Готово, мистер Олькофф."

    def test_ударения_для_piper(self, monkeypatch):
        monkeypatch.setattr(config, "STRESS_FIXES", {"добрыня": "добр+ыня"})
        assert voice.apply_stress_fixes("Добрыня на связи") == "Добр+ыня на связи"
        assert voice.plus_to_acute("добр+ыня") == "добры́ня"

    def test_числа_и_знаки_словами(self):
        spoken = voice._numbers_to_words("сейчас 14°, в 18:30 дождь, 90%")
        assert "четырнадцать" in spoken and "градусов" in spoken and "процентов" in spoken
        assert "°" not in spoken and "%" not in spoken
