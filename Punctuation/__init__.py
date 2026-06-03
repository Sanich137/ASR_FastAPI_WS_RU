# -*- coding: utf-8 -*-
from starlette.requests import HTTPConnection
import logging
import asyncio
import numpy as np
from transformers import AutoTokenizer
import onnxruntime as ort
from pathlib import Path
import pynvml

from typing import List
logger = logging.getLogger(__name__)

# Прогнозируемые знаки препинания
PUNK_MAPPING = {".": "PERIOD", ",": "COMMA", "?": "QUESTION"}

# Прогнозируемый регистр LOWER - нижний регистр, UPPER - верхний регистр для первого символа,
# UPPER_TOTAL - верхний регистр для всех символов
LABELS_CASE = ["LOWER", "UPPER", "UPPER_TOTAL"]
# Добавим в пунктуацию метку O означающий отсутствие пунктуации
LABELS_PUNC = ["O"] + list(PUNK_MAPPING.values())

# Сформируем метки на основе комбинаций регистра и пунктуации
LABELS_list = []

for case in LABELS_CASE:
    for punc in LABELS_PUNC:
        LABELS_list.append(f"{case}_{punc}")

LABELS = {label: i + 1 for i, label in enumerate(LABELS_list)}
LABELS["O"] = -100
INVERSE_LABELS = {i: label for label, i in LABELS.items()}

LABEL_TO_PUNC_LABEL = {
    label: label.split("_")[-1] for label in LABELS.keys() if label != "O"
}
LABEL_TO_CASE_LABEL = {
    label: "_".join(label.split("_")[:-1]) for label in LABELS.keys() if label != "O"
}


def token_to_label(token, label):
    if type(label) == int:
        label = INVERSE_LABELS[label]
    if label == "LOWER_O":
        return token
    if label == "LOWER_PERIOD":
        return token + "."
    if label == "LOWER_COMMA":
        return token + ","
    if label == "LOWER_QUESTION":
        return token + "?"
    if label == "UPPER_O":
        return token.capitalize()
    if label == "UPPER_PERIOD":
        return token.capitalize() + "."
    if label == "UPPER_COMMA":
        return token.capitalize() + ","
    if label == "UPPER_QUESTION":
        return token.capitalize() + "?"
    if label == "UPPER_TOTAL_O":
        return token.upper()
    if label == "UPPER_TOTAL_PERIOD":
        return token.upper() + "."
    if label == "UPPER_TOTAL_COMMA":
        return token.upper() + ","
    if label == "UPPER_TOTAL_QUESTION":
        return token.upper() + "?"
    if label == "O":
        return token


def decode_label(label, classes="all"):
    if classes == "punc":
        return LABEL_TO_PUNC_LABEL[INVERSE_LABELS[label]]
    if classes == "case":
        return LABEL_TO_CASE_LABEL[INVERSE_LABELS[label]]
    else:
        return INVERSE_LABELS[label]


class SbertPuncCaseOnnx:
    def __init__(self, onnx_model_path, use_gpu = False):
        self.sessions: List[ort.InferenceSession] = []

        self.tokenizer = AutoTokenizer.from_pretrained(onnx_model_path,
                                                       strip_accents=False,
                                                       )
        session_options = ort.SessionOptions()
        session_options.log_severity_level = 4  # Выключаем подробный лог
        session_options.enable_profiling = False
        session_options.enable_mem_pattern = False      # True в диаризации
        session_options.enable_mem_reuse = False        # True в диаризации
        session_options.enable_cpu_mem_arena = False    # True в диаризации
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.inter_op_num_threads = 0
        session_options.intra_op_num_threads = 0
        session_options.add_session_config_entry("session.disable_prepacking", "1")  # Отключаем дублирование весов
        session_options.add_session_config_entry("session.use_device_allocator_for_initializers", "1")

        if not use_gpu:
            providers = ['CPUExecutionProvider']
        else:
            providers = [('CUDAExecutionProvider', {
                'device_id': 0,
                'arena_extend_strategy': 'kSameAsRequested', # 20.657809 на 1000 итераций и 26 Мб съел
                'gpu_mem_limit': int(1.8 * 1024 * 1024 * 1024),  # Потребляет где-то 1.9 Гб памяти ГПУ
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
                'do_copy_in_default_stream': True,
            }),
                         'CPUExecutionProvider']

        model_pth = Path(onnx_model_path) / "model.onnx"

        # Todo - можно собрать  очередь сессий. Очень интересный механизм для оптимизации производительности
        self.session = ort.InferenceSession(path_or_bytes=model_pth,
                                    sess_options=session_options,
                                    providers=providers,
                                    logger=logger)


    async def punctuate(self, text):
        text = text.strip().lower()
        # Разобьем предложение на слова
        words = text.split()

        tokenizer_output = self.tokenizer(words, is_split_into_words=True)

        if len(tokenizer_output.input_ids) > 512:
            return " ".join(
                [
                    await self.punctuate(" ".join(text_part))
                    for text_part in np.array_split(words, 2)
                ]
            )

        # Подготовка входных данных для модели
        input_ids = np.array(tokenizer_output.input_ids, dtype=np.int64).reshape(1, -1)
        attention_mask = np.array(tokenizer_output.attention_mask, dtype=np.int64).reshape(1, -1)
        token_type_ids = np.zeros_like(input_ids, dtype=np.int64)  # Добавляем token_type_ids

        # Выполнение модели
        loop = asyncio.get_event_loop()
        outputs = await loop.run_in_executor(
            None,
            lambda: self.session.run(None, {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,  # Передаём token_type_ids
                        } ))

        predictions = np.argmax(outputs[0], axis=2)

        # decode punctuation and casing
        splitted_text = []
        word_ids = tokenizer_output.word_ids()
        for i, word in enumerate(words):
            label_pos = word_ids.index(i)
            label_id = predictions[0][label_pos]
            label = decode_label(label_id)
            splitted_text.append(token_to_label(word, label))
        capitalized_text = " ".join(splitted_text)
        return capitalized_text

    async def process_punctuation_sessions(self, text):
        try:
            capitalized_text = await self.punctuate(text)
        except Exception as e:
            logger.error(f"Ошибка пунктуатора - {e}")

        return capitalized_text

def gpu_stat(gpu_index):

    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)  # Первая видеокарта
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free_mb = mem_info.free / 1024**2
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_load = utilization.gpu
        temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
    except pynvml.NVMLError as e:
        return {"error": str(e)}
    finally:
        pynvml.nvmlShutdown()
    return free_mb, gpu_load,temperature


def get_punctuator(conn: HTTPConnection):
    return conn.app.state.punctuator


if __name__ == '__main__':
    from datetime import datetime as dt

    test_input_text = [
        "Живут грызуны по лесам и полям что они едят эти зверьки грызут зерна и кору деревьев зайцы обгладывают яблони в саду грызуны портят посевы",
        "Солнце только что встало небо ясное все вокруг блестит как хорошо на свежем воздухе слышишь пение жаворонка звонкий голосок слышен в ясной вышине",
        "В сырых местах живут неядовитые змейки у них два желтых пятна на затылке ужи любят воду и хорошо плавают кормятся ужи лягушками и рыбами вы видели ужа не бойтесь его",
        "Белка бойко лазит по деревьям какая она ловкая на ушах у белки кисточки хвост длинный и пушистый зачем он ей белка прикрывается хвостом от холода он служит ей рулём при прыжках",
        "Когда ты был в цирке вспомни яркие афиши и флажки вот жонглёр ловит на лету тарелки фокусник превратил шляпу в букет цветов а вот клоун он смешит людей у него в куртке петух как интересно в цирке",
        "Вот ночная хищная птица голова у нее круглая клюв крючком когти острые Узнали ее это сова она живет в лесах или на чердаках домов ночью птица ловит мышей",
        "Черепахи живут на земле и в воде они откладывают яйца прямо на камни черепахи не высиживают их яйца лопаются сами появляются маленькие черепашата а какого размера эти пресмыкающиеся черепахи бывают маленькие и очень большие",
        ]

    model_path = str(Path("../models/sbert_punc_case_ru_onnx"))
    print(f" ресурсы до старта приложения  {gpu_stat(0)}")
    time_start = dt.now()
    sbertpunc = SbertPuncCaseOnnx(model_path, use_gpu=True)
    print(f"Время на инициализацию {(dt.now() - time_start).total_seconds()}")
    print(f"Ресурсы после старта приложения  {gpu_stat(0)}")
    import logging as logger

    # for _ , text_ in enumerate(input_text*100):
    #     punctuated = asyncio.run(sbertpunc.process(text_)
    #                     )
    #     # print(punctuated)

    async def process_texts(texts):
        tasks = [sbertpunc.process_punctuation_sessions(text) for text in texts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results


    time_start = dt.now()
    texts = test_input_text * 100
    punctuated_texts = asyncio.run(process_texts(texts))
    # for text, punctuated in zip(texts, punctuated_texts):
    #     print(f"Punctuated: {punctuated}\n")



    print(f" ресурсы после окончания работы приложения  {gpu_stat(0)}")
    print(f"Время выполнения {(dt.now() - time_start).total_seconds()}")

### Увеличение количества сессий в адаптере приводит к пропорциональному увеличению расхода памяти (1,9 * Х для пунктуации) и где-то на 30% быстрее 7,676 против 5,579.
### Если использовать 2 ГПУ, то выполняется где-то на быстрее 60% 4.63946 против 7,676