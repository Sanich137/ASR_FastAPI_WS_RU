from utils.do_logging import logger
import numpy as np
import asyncio
from config import settings
from Punctuation import sbertpunc

async def do_sensitizing(input_asr_json: str, do_punctuation: bool = False):
    """
    :param do_punctuation: Если True, то производит пунктуацию и капитализацию над собранными в предложения выражения.
    :param input_asr_json: {"channel_{n_channel + 1}":
                                {"data":{"result":
                                            [
                                                { "conf": 1,
                                                    "start": float(),
                                                    "end": float(),
                                                    "word": str()
                                                }, ...
                                            ],
                                        "text": str()
                                },
                                ...
                            }
    :return:  {
            "raw_text_sentenced_recognition": str() - текст с переносами, сортировкой по времени начала спикера,
            "list_of_sentenced_recognitions": list() - список словарей с текстом, временем начала и номером спикера
            "full_text_only": text_only, list() - список с суммированным текстом каждого канала
            "err_state": err_state - ошибка, если не удалось
        }
    """

    err_state = None
    sentenced_recognition = []
    text_only = []

    if not input_asr_json:
        err_state = "Err_No_data_to_sensitize"
    else:
        for channel in input_asr_json:
            words = list()
            word_pause = 1.5
            sentence_element = None
            one_text_only = str()
            err_state = None

            for results in input_asr_json[channel]:
                if "result" in results["data"]:
                    for w in results["data"]["result"]:
                        words.append(w)
                    one_text_only += (results["data"].get("text", None)) + ' '
                else:
                    continue

            if len(words) < 1:
                err_state = f"Err_No_words in {channel}"
                sentence_element = []
                logger.info(f"В списке слов только одно слово в канале {channel}")
            else:
                sentence_element = []
                sentences = []
                start_time = words[0].get('start')
                end_time = words[0].get('end')

                if len(words) == 1:
                    # Обработка случая с одним словом
                    text = words[0].get('word')
                    if do_punctuation:
                        text = await sbertpunc.punctuate(text)

                    sentence_element.append({
                      "start": start_time,
                      "end": end_time,
                      "text": text,
                      "speaker": channel
                    })
                else:
                    # Оригинальная логика для нескольких слов
                    between_words_delta = []
                    end_time = words[0].get('end')

                    for word in words[1:]:
                        between_words_delta.append(word.get('end') - end_time)
                        end_time = word.get('end')

                    words_mean = np.percentile(between_words_delta, settings.BETWEEN_WORDS_PERCENTILE)
                    logger.debug(f"words_mean = {words_mean}")

                    start_time = 0
                    end_time = 0

                    for word in words:
                        logger.debug(f"В работе слово '{word.get('word')}' 'start'={word.get('start')}, 'end'={word.get('end')}, "
                                    f"start_time={start_time}, end_time={end_time}")

                        if start_time == 0:
                            start_time = word.get('start')
                        if end_time == 0:
                            end_time = word.get('end')

                        logger.debug(f"start_time={start_time}, end_time={end_time}")

                        if word.get('start') < end_time:
                            logger.debug(f"Ошибка при логике расчёта времени на слове {word.get('word')} start = {word.get('start')}, end = {end_time}")

                        if (word.get('start') - end_time) < words_mean:
                            sentences.append(word.get('word'))
                            end_time = word.get('end')
                            continue

                        else:
                            if do_punctuation:
                                text = await sbertpunc.punctuate(' '.join(str(word) for word in sentences))
                            else:
                                text = ' '.join(str(word) for word in sentences)

                            sentence_element.append({
                                "start": start_time,
                                "end": end_time,
                                "text": text,
                                "speaker": channel
                                })
                            # Переопределяем список
                            sentences = list()
                            # В новый список помещаем слово
                            sentences.append(word.get('word'))
                            start_time = word.get('start')
                            end_time = word.get('end')
                            continue

                    if do_punctuation:

                        text = await sbertpunc.punctuate(' '.join(str(word) for word in sentences))
                    else:
                        text = ' '.join(str(word) for word in sentences)

                    sentence_element.append({
                        "start": start_time,
                        "end": end_time,
                        "text": text,
                        "speaker": channel
                      })

            sentenced_recognition.append(sentence_element)

            if do_punctuation:
                # Тут он текст разделит сам.
                one_text_only = str(await sbertpunc.punctuate(one_text_only))

            text_only.append(one_text_only)

    sentenced_recognition_joined = [element for sentence in sentenced_recognition for element in sentence]
    new_sentenced_recognition = sorted(sentenced_recognition_joined, key=lambda d: d['start'])

    for speaker_sentence_index in reversed(range(len(sentenced_recognition))):
        for sentence in sentenced_recognition[speaker_sentence_index]:
            sentence['combined_text'] = sentence["speaker"] + ": " + sentence['text']

    raw_text_of_resentenced_recognition = str()
    for sentence in new_sentenced_recognition:
        raw_text_of_resentenced_recognition += sentence.get('combined_text') + '\n'

    for sentence in new_sentenced_recognition:
        sentence.pop('combined_text')

    return {
      "raw_text_sentenced_recognition": raw_text_of_resentenced_recognition,
      "list_of_sentenced_recognitions": new_sentenced_recognition,
      "full_text_only": text_only,
      "err_state": err_state
    }


if __name__ == "__main__":
    input_json = {
    "channel_1": [
      {
        "data": {
          "result": [
            {
              "conf": 1,
              "start": 2.32,
              "end": 2.56,
              "word": "алло"
            }
          ],
          "text": "алло"
        }
      },
      {
        "data": {
          "result": [
            {
              "conf": 1,
              "start": 19.86,
              "end": 19.9,
              "word": "ха"
            },
            {
              "conf": 1,
              "start": 20.18,
              "end": 20.26,
              "word": "нет"
            },
            {
              "conf": 1,
              "start": 20.42,
              "end": 20.62,
              "word": "вчера"
            },
            {
              "conf": 1,
              "start": 20.74,
              "end": 20.78,
              "word": "не"
            },
            {
              "conf": 1,
              "start": 20.9,
              "end": 21.14,
              "word": "видела"
            },
            {
              "conf": 1,
              "start": 21.34,
              "end": 21.82,
              "word": "сообщение"
            },
            {
              "conf": 1,
              "start": 21.94,
              "end": 21.98,
              "word": "ну"
            },
            {
              "conf": 1,
              "start": 22.1,
              "end": 22.7,
              "word": "доверенность"
            },
            {
              "conf": 1,
              "start": 22.82,
              "end": 22.86,
              "word": "да"
            },
            {
              "conf": 1,
              "start": 23.02,
              "end": 23.02,
              "word": "с"
            },
            {
              "conf": 1,
              "start": 23.14,
              "end": 23.3,
              "word": "собой"
            },
            {
              "conf": 1,
              "start": 23.46,
              "end": 23.74,
              "word": "возьмем"
            }
          ],
          "text": "ха нет вчера не видела сообщение ну доверенность да с собой возьмем"
        }
      },
      {
        "data": {
          "result": [
            {
              "conf": 1,
              "start": 31.82,
              "end": 32.02,
              "word": "ладно"
            },
            {
              "conf": 1,
              "start": 32.22,
              "end": 32.46,
              "word": "сейчас"
            },
            {
              "conf": 1,
              "start": 33.26,
              "end": 33.74,
              "word": "попробуем"
            },
            {
              "conf": 1,
              "start": 34.5,
              "end": 34.58,
              "word": "все"
            },
            {
              "conf": 1,
              "start": 34.82,
              "end": 35.06,
              "word": "хорошо"
            },
            {
              "conf": 1,
              "start": 35.22,
              "end": 35.46,
              "word": "поняла"
            },
            {
              "conf": 1,
              "start": 35.58,
              "end": 35.66,
              "word": "вас"
            }
          ],
          "text": " ладно сейчас попробуем все хорошо поняла вас"
        }
      },
      {
        "data": {
          "result": [
            {
              "conf": 1,
              "start": 35.98,
              "end": 36.38,
              "word": "спасибо"
            },
            {
              "conf": 1,
              "start": 36.98,
              "end": 37.02,
              "word": "да"
            },
            {
              "conf": 1,
              "start": 37.34,
              "end": 37.38,
              "word": "до"
            },
            {
              "conf": 1,
              "start": 37.5,
              "end": 37.86,
              "word": "свидания"
            }
          ],
          "text": "спасибо да до свидания"
        }
      }
    ],
    "channel_2": [
      {
        "data": {
          "result": [
            {
              "conf": 1,
              "start": 3.68,
              "end": 3.88,
              "word": "ольга"
            },
            {
              "conf": 1,
              "start": 4.04,
              "end": 4.48,
              "word": "вадимовна"
            },
            {
              "conf": 1,
              "start": 4.64,
              "end": 5.2,
              "word": "здравствуйте"
            },
            {
              "conf": 1,
              "start": 5.28,
              "end": 5.4,
              "word": "это"
            },
            {
              "conf": 1,
              "start": 5.48,
              "end": 5.84,
              "word": "компания"
            },
            {
              "conf": 1,
              "start": 6,
              "end": 6.56,
              "word": "сберправо"
            },
            {
              "conf": 1,
              "start": 6.88,
              "end": 7.32,
              "word": "подскажите"
            },
            {
              "conf": 1,
              "start": 7.44,
              "end": 7.88,
              "word": "пожалуйста"
            },
            {
              "conf": 1,
              "start": 8,
              "end": 8.08,
              "word": "вот"
            },
            {
              "conf": 1,
              "start": 8.2,
              "end": 8.24,
              "word": "мы"
            },
            {
              "conf": 1,
              "start": 8.36,
              "end": 8.44,
              "word": "вам"
            },
            {
              "conf": 1,
              "start": 8.52,
              "end": 8.72,
              "word": "вчера"
            },
            {
              "conf": 1,
              "start": 8.84,
              "end": 9.12,
              "word": "вечером"
            },
            {
              "conf": 1,
              "start": 9.2,
              "end": 9.2,
              "word": "в"
            },
            {
              "conf": 1,
              "start": 9.32,
              "end": 9.48,
              "word": "чате"
            },
            {
              "conf": 1,
              "start": 9.6,
              "end": 9.96,
              "word": "написали"
            },
            {
              "conf": 1,
              "start": 10.08,
              "end": 10.4,
              "word": "возможно"
            },
            {
              "conf": 1,
              "start": 10.52,
              "end": 10.56,
              "word": "вы"
            },
            {
              "conf": 1,
              "start": 10.68,
              "end": 10.72,
              "word": "не"
            },
            {
              "conf": 1,
              "start": 10.84,
              "end": 11.04,
              "word": "видели"
            },
            {
              "conf": 1,
              "start": 11.16,
              "end": 11.24,
              "word": "что"
            },
            {
              "conf": 1,
              "start": 11.4,
              "end": 11.4,
              "word": "в"
            },
            {
              "conf": 1,
              "start": 11.48,
              "end": 11.6,
              "word": "суд"
            },
            {
              "conf": 1,
              "start": 11.72,
              "end": 11.8,
              "word": "не"
            }
          ],
          "text": "ольга вадимовна здравствуйте это компания сберправо подскажите пожалуйста вот мы вам вчера вечером в чате написали возможно вы не видели что в суд не"
        }
      },
      {
        "data": {
          "result": [
            {
              "conf": 1,
              "start": 11.92,
              "end": 12.4,
              "word": "необходимо"
            },
            {
              "conf": 1,
              "start": 12.52,
              "end": 12.52,
              "word": "с"
            },
            {
              "conf": 1,
              "start": 12.64,
              "end": 12.84,
              "word": "собой"
            },
            {
              "conf": 1,
              "start": 12.96,
              "end": 13.2,
              "word": "взять"
            },
            {
              "conf": 1,
              "start": 13.52,
              "end": 14.16,
              "word": "доверенность"
            },
            {
              "conf": 1,
              "start": 14.28,
              "end": 14.28,
              "word": "и"
            },
            {
              "conf": 1,
              "start": 14.44,
              "end": 14.48,
              "word": "по"
            },
            {
              "conf": 1,
              "start": 14.6,
              "end": 15.16,
              "word": "возможности"
            },
            {
              "conf": 1,
              "start": 15.32,
              "end": 15.64,
              "word": "сделать"
            },
            {
              "conf": 1,
              "start": 15.76,
              "end": 16,
              "word": "копию"
            },
            {
              "conf": 1,
              "start": 16.16,
              "end": 16.44,
              "word": "данной"
            },
            {
              "conf": 1,
              "start": 16.56,
              "end": 17.2,
              "word": "доверенности"
            },
            {
              "conf": 1,
              "start": 21.72,
              "end": 21.76,
              "word": "да"
            }
          ],
          "text": "необходимо с собой взять доверенность и по возможности сделать копию данной доверенности да"
        }
      },
      {
        "data": {
          "result": [
            {
              "conf": 1,
              "start": 24.04,
              "end": 24.04,
              "word": "и"
            },
            {
              "conf": 1,
              "start": 24.2,
              "end": 24.48,
              "word": "копию"
            },
            {
              "conf": 1,
              "start": 24.64,
              "end": 25,
              "word": "сделайте"
            },
            {
              "conf": 1,
              "start": 25.16,
              "end": 25.6,
              "word": "пожалуйста"
            },
            {
              "conf": 1,
              "start": 25.72,
              "end": 25.76,
              "word": "по"
            },
            {
              "conf": 1,
              "start": 25.84,
              "end": 26.36,
              "word": "возможности"
            },
            {
              "conf": 1,
              "start": 26.52,
              "end": 26.6,
              "word": "это"
            },
            {
              "conf": 1,
              "start": 26.76,
              "end": 26.84,
              "word": "вот"
            },
            {
              "conf": 1,
              "start": 27.04,
              "end": 27.28,
              "word": "юрист"
            },
            {
              "conf": 1,
              "start": 27.4,
              "end": 27.8,
              "word": "попросил"
            },
            {
              "conf": 1,
              "start": 27.96,
              "end": 28.28,
              "word": "который"
            },
            {
              "conf": 1,
              "start": 28.4,
              "end": 28.56,
              "word": "будет"
            },
            {
              "conf": 1,
              "start": 28.8,
              "end": 29.08,
              "word": "который"
            },
            {
              "conf": 1,
              "start": 29.24,
              "end": 29.6,
              "word": "работает"
            },
            {
              "conf": 1,
              "start": 29.72,
              "end": 29.72,
              "word": "с"
            },
            {
              "conf": 1,
              "start": 29.84,
              "end": 30,
              "word": "вами"
            },
            {
              "conf": 1,
              "start": 30.32,
              "end": 30.56,
              "word": "хорошо"
            },
            {
              "conf": 1,
              "start": 34.52,
              "end": 34.56,
              "word": "да"
            },
            {
              "conf": 1,
              "start": 34.8,
              "end": 34.92,
              "word": "все"
            },
            {
              "conf": 1,
              "start": 35.16,
              "end": 35.44,
              "word": "хорошо"
            },
            {
              "conf": 1,
              "start": 35.6,
              "end": 35.64,
              "word": "да"
            }
          ],
          "text": "и копию сделайте пожалуйста по возможности это вот юрист попросил который будет который работает с вами хорошо да все хорошо да"
        }
      },
      {
        "data": {
          "result": [
            {
              "conf": 1,
              "start": 36.44,
              "end": 36.64,
              "word": "всего"
            },
            {
              "conf": 1,
              "start": 36.84,
              "end": 37.16,
              "word": "доброго"
            },
            {
              "conf": 1,
              "start": 37.32,
              "end": 37.36,
              "word": "до"
            },
            {
              "conf": 1,
              "start": 37.48,
              "end": 37.88,
              "word": "свидания"
            }
          ],
          "text": "всего доброго до свидания"
        }
      }
    ]
  }

    res = asyncio.run(do_sensitizing(input_asr_json=input_json), )
    print(res)
