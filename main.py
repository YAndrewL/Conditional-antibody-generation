# -*- coding: utf-8 -*-
# @Time         : 2022/5/10 18:13
# @Author       : Yufan Liu
# @Description  : Main for training process

from data import AntigenAntibodyPairedData
from model import Transformer
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import loggers as pl_loggers

import time
from params import GetParams

pl.seed_everything(42)

training_time = time.strftime('%m-%d-%H-%M-%S', time.localtime())

params = GetParams("./setting.yaml")
model_params = params.model_params()
trainer_params = params.trainer_params()
data_params = params.data_params()

model_params['experiment_tracking'] = {**trainer_params, **data_params}



data_module = AntigenAntibodyPairedData(**data_params)
model = Transformer(**model_params)

# logger and callbacks
tb_logger = pl_loggers.TensorBoardLogger(save_dir='experiment_outs/' + training_time, name='pl_logs', version='', default_hp_metric=False)
checkpoint = ModelCheckpoint(dirpath='experiment_outs/' + training_time + '/checkpoints', save_top_k=5, monitor='train_loss')
trainer_params['callbacks'] = [checkpoint]
trainer_params['logger'] = tb_logger


trainer = pl.Trainer(**trainer_params)


if __name__ == '__main__':
    start = time.time()
    trainer.fit(model, datamodule=data_module)
    print(f'Running time: {time.time() - start:2f}s.')
