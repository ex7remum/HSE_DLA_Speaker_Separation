import os
import gdown

if __name__ == "__main__":
    model_url = 'https://drive.google.com/uc?export=download&id=10uiHjhrpzlU9WsWsfGHVdwpauXhVTX1W'
    model_path = 'best_model/model.pth'
    if not os.path.exists(model_path):
        os.mkdir('best_model')
        print('Downloading SpexPlus model.')
        gdown.download(model_url, model_path)
        print('Downloaded SpexPlus model.')
    else:
        print('SpexPlus model already exists.')

    config_url = 'https://drive.google.com/uc?export=download&id=1h3CJnE7A0PbWeoE0a3EezhVgv97nlPG-'
    config_path = 'best_model/config.json'
    if not os.path.exists(config_path):
        print('Downloading SpexPlus model test config.')
        gdown.download(config_url, config_path)
        print('Downloaded SpexPlus model test config.')
    else:
        print('SpexPlus model config already exists.')