import json
import logging
from datetime import datetime
from pathlib import Path
from config import settings

logger = logging.getLogger(__name__)


async def save_to_file(json_data: json, file_name: Path = None):
    from utils.pre_start_init import paths

    file_extension = "md" if settings.HUMAN_FORMAT_MD_FILE else "json"
    output_folder = paths.get("result_local_recognition_folder")
    local_path = paths.get("local_recognition_folder")  # Получаем корневую папку для отслеживания

    if not file_name:
        # Если имя файла не передано - создаём в корне
        file_name = datetime.now().strftime("%Y%m%d_%H%M%S.md")
        file_path = output_folder / file_name
    else:
        # Преобразуем путь к относительному (от корневой папки)
        try:
            relative_path = file_name.relative_to(local_path)
        except ValueError:
            # Если файл не находится внутри корневой папки, сохраняем как есть
            relative_path = file_name.name

        # Собираем итоговый путь
        file_path = output_folder / relative_path

        # Меняем расширение
        file_path = file_path.with_suffix(f".{file_extension}")

        # Создаём папки, если их нет
        file_path.parent.mkdir(parents=True, exist_ok=True)

    # С
    if settings.HUMAN_FORMAT_MD_FILE:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"**Дата:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            try:
                if len(json_data.get("diarized_data", "").keys()) > 1:
                    list_full_text_only = json_data.get('sentenced_data', {}).get('list_of_sentenced_recognitions', [])
                    for sentences in list_full_text_only:
                        speaker = str(sentences.get('speaker', 'N/A'))
                        text = str(sentences.get('text', '')).strip()
                        f.write(f"Спикер_{speaker}: {text}\n")
                else:
                    texts = json_data.get('sentenced_data', {}).get('full_text_only', [])
                    for text in texts if isinstance(texts, list) else []:
                        f.write(str(text).strip() + "\n")
            except Exception as e:
                logger.error(f"Ошибка сохранения файла: {e}", exc_info=True)
                return False
            else:
                logger.info(f"Результат распознавания сохранён в файл {file_path}")
    else:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump({
                'raw_data': {'list_of_sentenced_recognitions': json_data.get('sentenced_data', {}).get(
                    'list_of_sentenced_recognitions', [])},
            }, f, ensure_ascii=False)