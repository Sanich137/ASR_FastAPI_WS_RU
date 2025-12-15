import asyncio

from numpy.ma.core import count
from sympy.physics.units import speed

from utils.do_logging import logger

# Парсим JSON
def process_asr_json_deprecated(input_json, time_shift = 0.0, multiplier=1):
    data = input_json
    # Формируем шаблон результата
    result = {"data": {"result": [], "text": ""}}

    # Собираем слова из токенов
    words = []
    current_word = {"tokens": [], "start": 0.0, "end": 0.0, "probs": []}

    if not data:
        return result

    for token, timestamp in zip(data["tokens"], data["timestamps"]):
        # Если токен начинается с пробела, это начало нового слова
        if token.startswith(" "):
            if current_word["tokens"]:
                words.append(current_word)
            current_word = {"tokens": [token.strip()], "start": timestamp, "end": timestamp}
        else:
            current_word["tokens"].append(token)
            current_word["end"] = timestamp

    # Добавляем последнее слово
    if current_word["tokens"]:
        words.append(current_word)


    for word in words:
        # Объединяем токены в слово
        word_text = "".join(word["tokens"]).strip()

        # Требуется в принимающем приложении. Для общей работы смысла не имеет.
        conf = 1

        # Добавляем слово в результат
        result["data"]["result"].append({
            "conf": round(conf, 2),
            "start": word.get("start", 0) +round(time_shift, 2),
            "end": word["end"]+round(time_shift, 2),
            "word": word_text
        })

        # Добавляем слово в итоговый текст
        result["data"]["text"] += word_text + " "

    # Убираем лишний пробел в конце текста
    result["data"]["text"] = result["data"]["text"].strip()

    logger.debug(f'Результат изменения формата ответа - {result}')

    return result


def process_gigaam_json_output(input_json, time_shift=0.0, multiplier=1):
    """

    :param input_json: Входящий результат распознавания
    :param time_shift: Время старта чанка от начала аудио
    :param multiplier: Коэффициент замедления аудио
    :return:
    """

    # Парсим JSON
    data = input_json
    logger.debug(f" на разбор после ASR получен JSON - {input_json}")
    # Формируем шаблон результата
    result = {"data": {"result": [], "text": ""}}

    # Порог для разделения слов по времени (в секундах)
    time_gap_threshold = 2.0

    # Собираем слова из токенов
    words = []
    current_word = ""
    current_timestamps = []


    for i, token in enumerate(data['tokens']):
        timestamp = round((data['timestamps'][i] * multiplier) + time_shift, 2) # (1/multiplier-1)

        if token != ' ':
            current_timestamps.append(timestamp)
            # Проверяем временной промежуток между текущим и предыдущим токеном
            if len(current_timestamps) > 1:
                time_gap = current_timestamps[-1] - current_timestamps[-2]
                if time_gap > time_gap_threshold:
                    # Завершаем текущее слово
                    if current_word:
                        words.append({
                            'word': current_word,
                            'start': current_timestamps[0],
                            'end': current_timestamps[-2]
                        })
                    # Начинаем новое слово с текущего токена
                    current_word = token
                    current_timestamps = [timestamp]
                    continue

            current_word += token
        else:
            # Пробел завершает текущее слово
            if current_word:
                words.append({
                    'word': current_word,
                    'start': current_timestamps[0],
                    'end': current_timestamps[-1]
                })
                current_word = ""
                current_timestamps = []

    # Добавляем последнее слово, если оно есть
    if current_word:
        words.append({
            'word': current_word,
            'start': current_timestamps[0],
            'end': current_timestamps[-1]
        })




    # Формируем итоговый массив
    result['data'] = {
        'result': [{'conf': 1.0, 'start': word['start'], 'end': word['end'], 'word': word['word']} for word in words],
        'text': ' '.join(word['word'] for word in words),  # Обновляем текст на основе разделённых слов
    }
    return result


if __name__ == "__main__":
    # asr_str_json = {'emotion': '', 'event': '', 'lang': '', 'text': 'амлекс точка ру екатерина здравствуйте скажитепожалуйса как можно к вам обращить', 'timestamps': [2.48, 2.64, 2.84, 2.88, 2.96, 3.04, 3.12, 3.16, 3.24, 3.28, 3.36, 3.4, 3.44, 3.52, 3.56, 3.64, 3.76, 3.84, 3.88, 3.96, 4.0, 4.08, 4.12, 4.16, 4.24, 4.28, 4.36, 4.4, 4.48, 4.52, 4.56, 4.6, 4.68, 4.72, 4.76, 4.8, 4.88, 4.92, 15.76, 15.88, 15.92, 15.96, 16.04, 16.08, 16.12, 16.16, 16.24, 16.32, 16.36, 16.4, 16.44, 16.48, 16.52, 16.56, 16.64, 16.72, 16.76, 16.8, 16.84, 16.88, 16.92, 16.96, 17.0, 17.04, 17.08, 17.12, 17.16, 17.2, 17.24, 17.28, 17.32, 17.4, 17.44, 17.48, 17.52, 17.6, 17.64, 17.68, 17.72, 17.76], 'tokens': ['а', 'м', 'л', 'е', 'к', 'с', ' ', 'т', 'о', 'ч', 'к', 'а', ' ', 'р', 'у', ' ', 'е', 'к', 'а', 'т', 'е', 'р', 'и', 'н', 'а', ' ', 'з', 'д', 'р', 'а', 'в', 'с', 'т', 'в', 'у', 'й', 'т', 'е', ' ', 'с', 'к', 'а', 'ж', 'и', 'т', 'е', 'п', 'о', 'ж', 'а', 'л', 'у', 'й', 'с', 'а', ' ', 'к', 'а', 'к', ' ', 'м', 'о', 'ж', 'н', 'о', ' ', 'к', ' ', 'в', 'а', 'м', ' ', 'о', 'б', 'р', 'а', 'щ', 'и', 'т', 'ь'], 'words': []}
    json = {'emotion': '', 'event': '', 'lang': '',
            'text': 'аллону давай да проверим вот я что то там тебе поговорил этого достаточно',
            'timestamps': [2.08, 2.12, 2.24, 2.28, 19.84, 19.88, 20.24, 20.48, 20.52, 20.6, 20.64, 20.72, 20.8, 20.92,
                           20.96, 21.04, 21.2, 21.24, 21.28, 21.36, 21.4, 21.48, 21.52, 21.56, 21.64, 21.72, 21.76,
                           21.84, 21.88, 21.96, 22.0, 22.08, 22.12, 22.16, 22.24, 22.32, 22.36, 22.4, 22.48, 22.52,
                           22.56, 22.6, 22.72, 22.76, 22.84, 22.88, 22.92, 23.0, 23.08, 23.2, 23.24, 23.28, 23.32, 23.4,
                           23.44, 23.52, 23.96, 24.12, 24.2, 24.24, 24.28, 24.32, 24.4, 24.56, 24.6, 24.64, 24.72,
                           24.76, 24.84, 24.88, 24.92, 25.0, 25.04],
            'tokens': ['а', 'л', 'л', 'о', 'н', 'у', ' ', 'д', 'а', 'в', 'а', 'й', ' ', 'д', 'а', ' ', 'п', 'р', 'о',
                       'в', 'е', 'р', 'и', 'м', ' ', 'в', 'о', 'т', ' ', 'я', ' ', 'ч', 'т', 'о', ' ', 'т', 'о', ' ',
                       'т', 'а', 'м', ' ', 'т', 'е', 'б', 'е', ' ', 'п', 'о', 'г', 'о', 'в', 'о', 'р', 'и', 'л', ' ',
                       'э', 'т', 'о', 'г', 'о', ' ', 'д', 'о', 'с', 'т', 'а', 'т', 'о', 'ч', 'н', 'о'], 'words': []}

    res = process_gigaam_json_output(json)
    print(res)