# -*- coding: utf-8 -*-
# @Time         : 2022/5/20 12:30
# @Author       : Yufan Liu
# @Description  : Experiment metrics, perplexity, NLL etc
import numpy as np
import torch
from torch import nn
from .sampling import amino_vocab


class Metrics(object):
    def __init__(self, model, loader=None):
        self.model = model
        self.loader = loader
        self.model.eval()

    def perplexity(self):
        raw_list = []
        # sequence perplexity
        true_target = self.loader[2]
        out, _ = self.model(self.loader)
        criterion = nn.CrossEntropyLoss(ignore_index=-2, reduction='none')
        batch_mean_loss = criterion(out.permute(0, 2, 1), true_target - 2)  # [batch, max_seq_length]

        for i in range(len(batch_mean_loss)):
            loss = batch_mean_loss[i][batch_mean_loss[i].nonzero().squeeze(1)].mean(-1)  # delete zeros
            raw_list.append(loss.item())  # mean loss for a sequence

        return raw_list

    def negative_log_likelihood(self, sequence, antigen):
        antigen = antigen.unsqueeze(0)
        sequence = 'B' + sequence
        aa_tokens = list(map(lambda x: amino_vocab[x], list(sequence)))

        nll = 0
        input_seq = []
        for i in range(len(aa_tokens) - 1):
            input_seq.append(aa_tokens[i])
            to_pred = aa_tokens[i+1]
            input_tensor = torch.tensor(input_seq).unsqueeze(0).long()
            out, _ = self.model.decoder(input_tensor, antigen, None, None)
            probs = torch.nn.Softmax(-1)(out)[0, -1, :].detach().numpy()
            nll += -np.log(probs[to_pred - 2])
        return nll





if __name__ == '__main__':
    """
    collator_fn = DataCollator()
    data = PairedBinder('test', '../experiments_data/test_split/1OSP_O_abag.pkl')
    loader = next(iter(DataLoader(data, batch_size=100000, collate_fn=collator_fn)))

    model_params = GetParams("./setting.yaml").model_params()
    model = Transformer.load_from_checkpoint("./experiment_outs/epoch=424-step=40800.ckpt", **model_params)
    perp = Perplexity(model, loader)

    perp = perp.compute()
    print(max(perp), min(perp), np.mean(perp))
    """
    pass
