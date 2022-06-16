# -*- coding: utf-8 -*-
# @Time         : 2022/5/20 12:37
# @Author       : Yufan Liu
# @Description  : Experiments from experiments track
from functools import partial

from experiment_track import CDRGenerate, Metrics
from data import DataCollator, PairedBinder
from torch.utils.data import DataLoader
from model import Transformer
from params import GetParams
from typing import Dict
import numpy as np
import os
from transformers import BertTokenizer, BertModel
import time


def model_loader(ckpt, model_params: Dict):
    return Transformer().load_from_checkpoint(ckpt, **model_params)


if __name__ == '__main__':
    collator_fn = DataCollator()
    params = GetParams("./setting.yaml").model_params()
    current_model = model_loader("./experiment_outs/epoch=969-step=93120.ckpt", params)

    tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False,
                                              cache_dir='../absolute_antibody/cache_model')
    pretrained_model = BertModel.from_pretrained("Rostlab/prot_bert",
                                                 cache_dir='../absolute_antibody/cache_model')
    antigen_seq = 'KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNSQATNRNTDGSTDYGVLQINSRWWCND' \
                  'GRTPGSRNLCNIPCSALQSSDITATANCAKKIVSDGNGMNAWVAWRKHCKGTDVRVWIKGCRL'
    antigen_seq = ' '.join(list(antigen_seq))
    antigen_encode = pretrained_model(**tokenizer(antigen_seq,
                                                  return_tensors='pt')).last_hidden_state.squeeze(0)[1:-1, :].detach()
    # A
    """
    # perplexity exp.
    for i in os.listdir("../experiments_data/test_split"):
        data = PairedBinder('test', "../experiments_data/test_split/" + i)
        loader = next(iter(DataLoader(data, batch_size=1, collate_fn=collator_fn)))
        metrics = Metrics(model=current_model, loader=loader)
        mean_loss = metrics.perplexity()
        perp = np.exp(mean_loss)
        mean_perp = np.exp(np.mean(mean_loss))
        print(f"PDB: {i[:6]}, Max perp: {np.max(perp):4f}, Mean perp: {np.min(perp):4f}, Average perp: {np.mean(mean_perp):4f}")
    """

    # B

    # s = time.time()
    # generator = CDRGenerate(model=current_model,
    #                         antigen=antigen_encode)
    # print(generator.generate(1, 40, p=0.9, method='top-p'))
    # print(f"Running time: {time.time() - s}s.")

    # C
    metrics = Metrics(model=current_model)
    print(metrics.negative_log_likelihood("FYLRGLLLLHVW", antigen=antigen_encode))
