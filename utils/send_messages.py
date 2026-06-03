import logging
logger = logging.getLogger(__name__)

async def send_messages(_socket, _channel_name = None, _data=None, _silence=True, _error=None, _last_message=False,
                        _sentenced_data=None):
    if _sentenced_data is None:
        _sentenced_data = dict()
    ws = _socket
    is_ok = False
    if _last_message:
        logger.debug('Последнее сообщение')
    if not _data:
        data = ""
    else:
        data = _data.get('data')

    snd_mssg = {"channel_name": _channel_name,
                "silence": _silence,
                "data": data,
                "error": _error,
                "last_message": _last_message,
                "sentenced_data": _sentenced_data,
                }
    try:
        await ws.send_json(snd_mssg)
    except Exception as e:
        logger.error(f"send_message - exception - {e}")
    else:
        logger.debug(snd_mssg)
        logger.info(snd_mssg)
        is_ok = True

    return is_ok