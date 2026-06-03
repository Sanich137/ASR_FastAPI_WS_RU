# import trash.test_data
from typing import List, Dict, Any
from difflib import SequenceMatcher
import logging
logger = logging.getLogger(__name__)



def are_words_similar(word1: str, word2: str, similarity_threshold: float = 0.8) -> bool:
    """
    Проверяет, похожи ли два слова на основе коэффициента схожести.
    :param: word1: Первое слово.
    :param: word2: Второе слово.
    :param: similarity_threshold: Порог схожести (от 0 до 1).
    :return: True, если слова похожи, иначе False.
    """

    similarity = SequenceMatcher(None, word1, word2).ratio()
    if similarity >= similarity_threshold:
        if similarity != 1:
            logger.debug(f"==== слово '{word1}' похоже на слово '{word2}'")
        return True


async def remove_echo(input_json: Dict[str, List[Dict[str, Any]]], delta: float = 0.9) -> Dict[str, List[Dict[str, Any]]]:
    output_json = {}

    # Получаем список каналов
    channels = list(input_json.keys())

    # Создаём копию входных данных для изменений
    for channel_name in channels:
        output_json[channel_name] = [msg.copy() for msg in input_json[channel_name]]

    # Проходим по всем каналам
    for i, channel_name in enumerate(channels):
    # for i, channel_name in enumerate(reversed(channels)):
        for message in output_json[channel_name]:
            if not message.get("silence", False):  # Игнорируем тишину
                # Создаём список для хранения слов, которые не являются эхом
                cleaned_words = []

                for word in message["data"]["result"]:
                    is_echo = False

                    # Проверяем другие каналы
                    for other_channel_name in channels:
                        if other_channel_name != channel_name:  # Игнорируем текущий канал
                            for other_message in output_json[other_channel_name]:
                                if not other_message.get("silence", False):  # Игнорируем тишину
                                    for other_word in other_message["data"]["result"]:
                                        # Проверяем, находится ли слово в пределах дельты до или после текущего слова
                                        if abs(word["end"] - other_word["end"]) <= delta and \
                                            are_words_similar(word["word"], other_word["word"]):
                                            # Если слово найдено до текущего, удаляем его из текущего канала
                                            if other_word["end"] < word["end"]: # Оставляем, если сдвиг слишком мал
                                                is_echo = True

                                                logger.debug(f"<--- слово '{word['word']}' найдено до текущего, удаляем '{word['word']}' из текущего канала")
                                                logger.debug(f"В текущем канале время {word['start']}")
                                                logger.debug(f"В другом канале время {other_word['start']}")


                                                break
                                            # Если слово найдено после текущего, удаляем его из другого канала
                                            elif other_word["end"] > word["end"]:
                                                other_message["data"]["result"].remove(other_word)
                                                other_message["data"]["text"] = " ".join(
                                                    [w["word"] for w in other_message["data"]["result"]]
                                                )
                                                logger.debug(
                                                    f"---> слово '{word['word']}' найдено после текущего, удаляем '{other_word['word']}' из другого канала")
                                                logger.debug(f"В текущем канале время {word['start']}")
                                                logger.debug(f"В другом канале время {other_word['start']}")

                                    if is_echo:
                                        break
                            if is_echo:
                                break

                    # Если слово не является эхом, добавляем его в очищенный список
                    if not is_echo:
                        cleaned_words.append(word)

                # Обновляем результат и текст в текущем сообщении
                message["data"]["result"] = cleaned_words
                message["data"]["text"] = " ".join([w["word"] for w in cleaned_words])
        # break

    return output_json



# if __name__ == "__main__":
#
#     input_data = trash.test_data.for_echo["raw_data"]
#     print(remove_echo(input_data))




