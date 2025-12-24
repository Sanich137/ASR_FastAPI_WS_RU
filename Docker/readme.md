# Инструкция по установке и запуске контейнеров ASR.
- Файл [Dockerfile](Dockerfile) создан для запуска как на GPU Nvidia, так и на CPU.
- Выбор в основе докер лежит простой образ python:3.12-slim-bullseye, а поддержка CUDA обеспечивается onnxruntime-gpu cо встроенными библиотеками.
- В образе уже будет установлены модели SileroVAD и voxblink2_samresnet100_ft для диаризации.
- При первом запуске будет скачана модель для ASR. По умолчанию это gigaam-v3-ctc. 
- Важно! Чтобы модель не перекачивалась при каждом запуске, присоедините каталог models/hub к локальному.
- Для замены voxblink2_samresnet100_ft при старте передайте её название переменной окружения. Список в [read.me](readme.md)
- Для замены модели ASR - передайте её название переменной окружения.  Список в [read.me](readme.md)
- для выбора провайдера доступны CPU, CUDA, TENSORRT. Последний существенно увеличивает скорость, и объем контейнера.

## Настройка оборудования для работы CUDA. (Для контейнера _CPU пропускаем).
- Установите [Nvidia container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), чтобы драйвер видеокарты был доступен в контейнере. 

## Сборка и запуск контейнера.
- Скачиваем необходимый докер файл и помещаем его в любую папку.
- - Переходим в папку и создаём образ, например для CUDA:
```bash
docker build -t asr -f /mnt/e/Coding/Docker/Dockerfile /mnt/e/Coding/Docker --build-arg PROVIDER=CUDA --build-arg PROVIDER=CUDA    
```
 
- Запускаем образ c GPU:
```commandline
docker run --runtime=nvidia -it --rm -p 8888:49153 asr 
           --runtime=nvidia --gpus all -p 8888:49153 -v f:\models\hub\:/ASR_FastAPI_WS_RU/models/hub/


```
- Запускаем образ на CPU:
```commandline
docker run -it --rm -p 8888:49153 asr
```

## Запуск с локальным распознаванием 
По примерам выше Вы получите классический API сервер.
Но Проект умеет распознавать и локально предоставленные файлы.
Для этого нам нужно передать при старте переменную DO_LOCAL_FILE_RECOGNITIONS=1 и DELETE_LOCAL_FILE_AFTR_ASR=1, если мы
хотим чтобы сервис сам удалял отработанные аудиофайлы. И указать папки, в которые мы будем помещать файлы для 
распознавания и получать результаты распознавания. 

выглядеть строка запуска будет так:

```bash
docker run -d 
  --rm \
  -p 8888:49153 \
  -v /путь/на/хосте/logs:/ASR_FastAPI_WS_RU_sherpa-onnx/logs \
  -v /путь/на/хосте/to_asr:/ASR_FastAPI_WS_RU_sherpa-onnx/local_asr/to_asr \
  -v /путь/на/хосте/after_asr:/ASR_FastAPI_WS_RU_sherpa-onnx/local_asr/after_asr \
  -e DO_LOCAL_FILE_RECOGNITIONS=1 \
  -e DELETE_LOCAL_FILE_AFTR_ASR=1 \
  asr

```
Помещаете файлы в папку to_asr, результат отобразится через некоторое время в папке after_asr.

## Изменение параметров при сборке и запуске
При сборке и запуске контейнера можно передать значение переменных виртуального окружения. Например, если Вы хотите 
использовать для диаризации другую модель, например "voxceleb_gemini_dfresnet114_LM" (полный список есть в основном ридми)
то при сборке контейнера и при его запуске необходимо передать дополнительный параметр -e DIAR_MODEL_NAME=:
```bash
docker build \
  --build-arg DIAR_MODEL_NAME=voxceleb_gemini_dfresnet114_LM \ 
  -t  \
  asr /path/to/Dockerfile_GigaAM
```
В таком случае, обязательно и при запуске контейнера передать название модели диаризации. В противном случае, при каждом 
запуске будет скачиваться вновь установленная по умолчанию **voxblink2_samresnet100_ft**:

```bash
docker run -it --rm -e DIAR_MODEL_NAME=voxceleb_gemini_dfresnet114_LM -p 8888:49153 asr
```
Все остальные параметры виртуального окружения можно передать при запуске контейнера без необходимости передавать их при сборке.
Полный список тоже смотрите в основном ридми.


- Для запуска в Pycharm так же не забываем указать эти дополнительыне  **Run options**:  "--runtime=nvidia -p 8888:49153"

- Сервис будет доступен по адресу http://127.0.0.1:8888/docs#
- Страница для тестов будет доступна по адресу http://127.0.0.1:8888/demo

Работоспособность проверена на Win11 wsl2, Pycharm Win11.