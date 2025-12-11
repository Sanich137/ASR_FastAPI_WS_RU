import datetime
from datetime import datetime as dt
from turtledemo.penrose import start

import ujson
import numpy as np
from collections import defaultdict

from sympy.physics.units import steradian

import config
from Recognizer import recognizer
from utils.bytes_to_samples_audio import get_np_array_samples_float32
from utils.resamppling import resample_audiosegment
from utils.slow_down_audio import do_slow_down_audio
from utils.do_logging import logger
from dataclasses import asdict
from onnx_asr.utils import read_wav_files, pad_list



def calc_speed(data):
    time_to_speak_tokens = 0
    count_tokens = len(data["tokens"]) - data["tokens"].count(' ')
    if count_tokens == 0:
        return 0

    ind = [i for i, val in enumerate(data["tokens"]) if val == " "]
    try:
        time_to_speak_tokens = (data["timestamps"][-1] -
                                sum([data["timestamps"][index+1] - data["timestamps"][index]
                                for index, token in enumerate(data["timestamps"]) if
                                index in ind and index < len(data["timestamps"])-1]))
    except Exception as e:
        logger.error(e)
    else:
        logger.debug(f"------------------------------")
        logger.debug(f"Всего токенов: {count_tokens}")
        logger.debug(f"Время на произношение токенов: {time_to_speak_tokens}")

    if time_to_speak_tokens != 0:
        speech_speed = count_tokens // time_to_speak_tokens
    else:
        speech_speed = 0

    logger.debug(f"Скорость речи: {speech_speed}")
    logger.debug(f"------------------------------")

    return speech_speed


# Ещё используется
async def recognise_w_calculate_confidence(audio_data,
                                     num_trials = 1,
                                     time_tolerance: float = 0.5) -> dict:
    """
    Вычисляет уверенность (`conf`) для каждого слова на основе многократного распознавания.

    :param audio_data: Аудиоданные в формате Audiosegment (puDub).
    :param num_trials: Количество попыток распознавания.
    :param time_tolerance: Допустимая погрешность временных меток (в секундах).
    :return: json c параметрами conf
    """

    # Результаты всех попыток распознавания
    result = dict()
    all_tokens = list()

    # Многократное распознавание
    for _ in range(num_trials):
        stream = recognizer.create_stream()
        # перевод в семплы для распознавания.
        samples = await get_np_array_samples_float32(audio_data.raw_data, 2)
        # передали аудиофрагмент на распознавание

        stream.accept_waveform(sample_rate=audio_data.frame_rate, waveform=samples)
        recognizer.decode_stream(stream)

        result = ujson.loads(str(stream.result))

        timestamps = result['timestamps']
        tokens = result['tokens']

        all_tokens.append(list(zip(tokens, timestamps)))

    del stream
    # Словарь для хранения статистики по токенам
    token_stats = defaultdict(lambda: {"count": 0, "variants": defaultdict(int)})

    # Анализируем результаты всех прогонов
    for trial in all_tokens:
        for token, start in trial:
            # Группируем токены по времени начала с допуском
            time_key = round(start / time_tolerance) * time_tolerance
            token_stats[(token, time_key)]["count"] += 1
            token_stats[(token, time_key)]["variants"][token] += 1

    # Формируем итоговые токены с conf
    tokens = []
    timestamps = []
    probs = []

    for (token, time_key), stats in token_stats.items():
        # Наиболее часто встречающийся вариант
        most_common_token = max(stats["variants"].items(), key=lambda x: x[1])[0]

        # Уверенность (conf)
        conf = stats["variants"][most_common_token] / num_trials

        tokens.append(most_common_token)
        timestamps.append(time_key)
        probs.append(round(conf, 4))

    result["tokens"] = tokens
    result["timestamps"]=timestamps
    result["probs"] =probs

    # Возвращаем результат в формате JSON
    return result


async def simple_recognise(audio_data, ) -> dict:
    # Приводим фреймрейт к фреймрейту модели
    if audio_data.frame_rate != config.BASE_SAMPLE_RATE:
        audio_data = await resample_audiosegment(audio_data, config.BASE_SAMPLE_RATE)

    # Перевод в семплы для распознавания.
    samples = await get_np_array_samples_float32(audio_data.raw_data, audio_data.sample_width)
    result = asdict(recognizer.recognize(samples, sample_rate=config.BASE_SAMPLE_RATE))

    return result


async def recognise_w_speed_correction(audio_data, multiplier=float(1.0), can_slow_down = False,
                                       ) -> tuple:
    """
    Распознавание чанка с возможностью контроля быстрой речи.

    :param multiplier: Float
    :param can_slow_down: Boolean
    :param audio_data: Аудиоданные в формате Audiosegment (puDub).
    :return: Dict
    """

    speed = 0
    if can_slow_down and multiplier < 1:
        logger.debug(f"Обработка быстрой речи с коэффициентом: {multiplier:.2f}")
        audio_data = await do_slow_down_audio(audio_segment=audio_data,slowdown_rate=multiplier)


    # Парсим результат
    result = await simple_recognise(audio_data)

    if can_slow_down and multiplier == 1:
        speed = calc_speed(result)
        logger.debug(f"Скорость аудио {speed} единиц в секунду")
        if speed > config.SPEECH_PER_SEC_NORM_RATE:
            # print(max((config.SPEECH_PER_SEC_NORM_RATE - 1) / speed, 0.8))

            result, speed, multiplier = await recognise_w_speed_correction(audio_data=audio_data,
                                               can_slow_down=True,
                                               multiplier=max((config.SPEECH_PER_SEC_NORM_RATE-1)/speed, 0.8)
                                                                           )
    return result, speed, multiplier


async def simple_recognise_batch(list_audio_data: list, batch_size: int = 8) -> list:
    logger.info(f"Выполняется батчинг с размером {batch_size}")
    start_recognition = datetime.datetime.now()

    list_of_samples = []
    list_of_lens = []
    max_samples = int(config.MAX_OVERLAP_DURATION * config.BASE_SAMPLE_RATE)

    # Перевод в семплы и выравнивание по длине
    for audio_data in list_audio_data:
        samples = await get_np_array_samples_float32(audio_data.raw_data, audio_data.sample_width)

        # Выравнивание до максимальной длины
        if len(samples) < max_samples:
            # Дополнение тишиной (нулями) до нужной длины
            padded_samples = np.zeros(max_samples, dtype=np.float32)
            padded_samples[:len(samples)] = samples
            list_of_samples.append(padded_samples)
            list_of_lens.append(padded_samples)
        elif len(samples) > max_samples:
            # Обрезка до максимальной длины
            logger.warning(f"Аудио длиной {len(samples) / config.BASE_SAMPLE_RATE:.2f} сек. "
                           f"превышает MAX_OVERLAP_DURATION ({config.MAX_OVERLAP_DURATION} сек.). "
                           f"Будет обрезано до {max_samples} семплов.")
            list_of_samples.append(samples[:max_samples])
            list_of_lens.append(len(samples[:max_samples]))

        else:
            # Идеальный размер
            list_of_samples.append(samples)
            list_of_lens.append(samples)

    waveform = *pad_list(list_of_samples), config.BASE_SAMPLE_RATE
    resampled = recognizer.resampler(*waveform)

    # Делаем препроцессинг на все данные сразу
    start_preprocess = datetime.datetime.now()
    preprocessed = recognizer.asr._preprocessor(*resampled)

    list_of_encoded_data = list()
    X, y = preprocessed
    n_samples = X.shape[0]

    # Вычисляем количество батчей
    n_batches = int(np.ceil(n_samples / batch_size))

    # Разбиваем массивы
    X_batches = np.array_split(X, n_batches)
    y_batches = np.array_split(y, n_batches)

    # Объединяем в кортежи
    preprocessed = list(zip(X_batches, y_batches))
    print(f"Препроцессинг за {(datetime.datetime.now() - start_preprocess).total_seconds()} секунд.")

    start_encoding = datetime.datetime.now()
    for preprocessed_batch in preprocessed:
        list_of_encoded_data.append(recognizer.asr._encode(*preprocessed_batch))
    print(f"Энкодинг за {(datetime.datetime.now() - start_encoding).total_seconds()} секунд.")

    x_merged_encoded_data = np.concatenate([X for X, _ in list_of_encoded_data], axis=0)
    y_merged_encoded_data = np.concatenate([y for _, y in list_of_encoded_data], axis=0)
    encoded = (x_merged_encoded_data, y_merged_encoded_data)

    decoded_list = list(recognizer.asr._decoding(*encoded, None))
    result = (recognizer.asr._decode_tokens(*tup_decoded) for tup_decoded in decoded_list)

    list_of_dict_result = [asdict(res) for res in result]
    print(f"Полное распознавание за {(datetime.datetime.now() - start_recognition).total_seconds()} секунд.")

    return list_of_dict_result




if __name__ == "__main__":
    json = {
      "lang" : "",
      "emotion" : "",
      "event" : "",
      "text" : "я буду говорить тридцати секундными интервалами каждый раз повышая свою скорость на двадцать слов в минуту первый интервал который кстати уже начался я произнесу сорок слов",
      "timestamps" : [ 0.64, 0.76, 0.92, 0.96, 1.08, 1.12, 1.24, 1.44, 1.48, 1.6, 1.64, 1.72, 1.76, 1.84, 1.92, 2.0, 2.2, 2.28, 2.32, 2.44, 2.52, 2.56, 2.68, 2.72, 2.92, 3.08, 3.16, 3.32, 3.36, 3.48, 3.6, 3.8, 3.84, 3.96, 4.0, 4.08, 4.32, 4.36, 4.48, 4.52, 4.6, 4.72, 4.8, 5.0, 5.08, 5.2, 5.24, 5.32, 5.68, 5.72, 5.84, 5.92, 5.96, 6.04, 6.16, 6.28, 6.32, 6.4, 6.52, 6.72, 6.8, 6.92, 7.0, 7.16, 7.2, 7.4, 7.56, 7.72, 7.8, 7.84, 7.96, 8.08, 8.24, 8.36, 8.4, 8.56, 8.64, 8.72, 8.8, 8.84, 8.96, 9.12, 9.16, 9.24, 9.4, 9.56, 9.6, 9.72, 9.84, 9.88, 9.96, 10.0, 10.12, 10.28, 10.36, 10.4, 10.52, 10.68, 10.76, 10.84, 10.96, 11.0, 11.16, 11.2, 11.36, 11.44, 11.76, 12.08, 12.16, 12.2, 12.32, 12.4, 12.44, 12.52, 12.68, 12.72, 12.8, 12.88, 12.92, 13.04, 13.12, 13.24, 13.36, 13.56, 13.6, 13.68, 13.76, 13.84, 13.88, 13.96, 14.04, 14.2, 14.28, 14.4, 14.44, 14.6, 14.64, 14.8, 14.96, 15.04, 15.08, 15.2, 15.4, 15.44, 15.6, 15.64, 15.76, 15.88, 15.96, 16.12, 16.24, 16.28, 16.56, 16.64, 16.72, 16.8, 16.92, 17.04, 17.08, 17.24, 17.28, 17.48, 17.72, 17.76, 17.92, 17.96, 18.08, 18.2, 18.32, 18.4, 18.44, 18.52 ],
      "tokens" : [ "я", " ", "б", "у", "д", "у", " ", "г", "о", "в", "о", "р", "и", "т", "ь", " ", "т", "р", "и", "д", "ц", "а", "т", "и", " ", "с", "е", "к", "у", "н", "д", "н", "ы", "м", "и", " ", "и", "н", "т", "е", "р", "в", "а", "л", "а", "м", "и", " ", "к", "а", "ж", "д", "ы", "й", " ", "р", "а", "з", " ", "п", "о", "в", "ы", "ш", "а", "я", " ", "с", "в", "о", "ю", " ", "с", "к", "о", "р", "о", "с", "т", "ь", " ", "н", "а", " ", "д", "в", "а", "д", "ц", "а", "т", "ь", " ", "с", "л", "о", "в", " ", "в", " ", "м", "и", "н", "у", "т", "у", " ", "п", "е", "р", "в", "ы", "й", " ", "и", "н", "т", "е", "р", "в", "а", "л", " ", "к", "о", "т", "о", "р", "ы", "й", " ", "к", "с", "т", "а", "т", "и", " ", "у", "ж", "е", " ", "н", "а", "ч", "а", "л", "с", "я", " ", "я", " ", "п", "р", "о", "и", "з", "н", "е", "с", "у", " ", "с", "о", "р", "о", "к", " ", "с", "л", "о", "в" ],
      "words" : [ ]
    }