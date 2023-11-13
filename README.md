# Speaker Separation project
## Overview
Репозиторий для обучения Speaker Separation модели на Librispeech датасете. 
Была использована архитектура SpexPlus. 
Получившийся Si-Sdr на кастомном датасете: 5.67, PESQ: 1.34. На публичном
датасете: Si-Sdr - 5.59, PESQ - 1.32.

**Важно**: если вы хотите самостоятельно обучить модель
нужно изменить
переменную split_dir в _create_index в hw_ss/datasets/librispeech_dataset.py
на путь до вашего датасета.


## Installation guide
Устанавливаются нужны библиотеки, а также скачиваeтся
получившийся чекпойнт. 
```shell
cd HSE_DLA_Speaker_Separation
pip install -r ./requirements.txt
python3 get_model.py
```

## Training model
Код для запуска обучения модели.

Флаги в квадратных скобках не использовались при обучении, но
при желании их можно использовать.

Значения флагов:

**-r** - путь до чекпойнта, если хотите продолжить обучение модели

**-t** - путь до тестового датасета

**-o** - путь до файла, куда будет записываться результат  
```shell
cd HSE_DLA_Speaker_Separation
python3 train.py -c hw_ss/configs/config.json
                 [-r default_test_model/checkpoint.pth]
                 [-t test_data]
                 [-o test_result.json]
```

## Testing
Код для тестирования скачанного чекпойнта. После того, как код выполнится,
в консоль будут выведены полученные метрики (Si-Sdr и PESQ).

Если хотите протестировать на своем датасете, нужно, чтобы он состоял
из трех папок: mix, refs и targets, в которых лежат файлы формата
<имя_файла>-<mixed/ref/target>.wav соответственно.

Флаги в квадратных скобках не использовались при обучении, но
при желании их можно использовать.
```shell
cd HSE_DLA_Speaker_Separation
python3  -r /kaggle/working/HSE_DLA_Speaker_Separation/best_model/model.pth \
         [-c your config if needed]
         [-t test_data]
         [-o test_result.json]
```

## Author
Юрий Максюта, БПМИ203