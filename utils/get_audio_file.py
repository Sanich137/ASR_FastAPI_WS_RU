import os
import logging
from typing import Any, Coroutine

import httpx
import io

from pydub import AudioSegment

from utils.pre_start_init import paths

async def getting_audiofile(file_url, post_id) -> tuple[bool, str, io.BytesIO | None]:
    res = False
    error = str()
    file_ext = file_url.path.split('/')[-1].split('.')[-1]
    buffer = None

    if file_ext in ['mp3', 'wav', 'ogg']:
        try:
            get_file_url = file_url.unicode_string()
        except Exception as e:
            logging.error(f"Error_url_parsing = {e}")
            return False, f"Error_url_parsing = {e}", None
        else:
            with httpx.Client() as sess:
                try:
                    response = sess.get(url=get_file_url)
                    file_data = response.content
                except Exception as e:
                    logging.error(f'Ошибка получения файла из ЕРП - {e}')
                    error = f"Getting file error - {e}"
                else:
                    buffer = io.BytesIO(file_data)
                    buffer.seek(0)
                    res = True
    else:
        error = "No audio file in request link"

    return res, error, buffer

# Todo - убрать корягу ниже
async def open_default_audiofile(post_id) -> tuple[bool, str, io.BytesIO | None]:
    res = False
    error_description = str()
    file = paths.get('test_file')
    file_ext = str(file).split('/')[-1].split('.')[-1]
    buffer = None

    if file_ext in ['mp3', 'wav', 'ogg']:
        try:
            buffer = io.BytesIO()
            AudioSegment.from_file(file=file).export(buffer, format="wav")
            buffer.seek(0)
            res = True
        except Exception as e:
            logging.error(f"Error_file_opening = {e}")
            error_description = str(e)
    else:
        error_description = "No audio file in request link"

    return res, error_description, buffer
