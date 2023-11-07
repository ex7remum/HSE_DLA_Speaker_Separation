# ASR project
## Overview
Репозиторий для обучения ASR модели на Librispeech датасете. 
Была использована архитектура DeepSpeech2. 
Получившийся WER на test-other: 28.8.

**Важно**: если вы хотите самостоятельно обучить модель или протестировать полученную
нужно изменить
переменную split_dir в _create_index в hw_asr/datasets/librispeech_dataset.py
на путь до вашего датасета.


## Installation guide
Устанавливаются нужны библиотеки, а также скачиваются языковые модели 
и получившийся чекпойнт. 
```shell
cd HSE_DLA_ASR
pip install -r ./requirements.txt
python3 prepare_lm.py 
python3 get_model.py
```

## Training model
Флаги в квадратных скобках не использовались при обучении, но
при желании их можно использовать. 
```shell
cd HSE_DLA_ASR
python3 train.py -c hw_asr/configs/config.json
                 [-r default_test_model/checkpoint.pth]
                 [-t test_data]
                 [-o test_result.json]
```

## Testing
Код для тестирования скачанного чекпойнта. После того, как код выполнится,
в консоль будут выведены полученные метрики, а также появится файл output.json
с примерами настоящих и предсказанных для них текстов.
Флаги в квадратных скобках не использовались при обучении, но
при желании их можно использовать.
```shell
cd HSE_DLA_ASR
python3  -r /kaggle/working/HSE_DLA_ASR/best_model/model.pth \
         [-c your config if needed]
         [-t test_data]
         [-o test_result.json]
```

## Author
Юрий Максюта, БПМИ203