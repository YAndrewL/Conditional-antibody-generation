# -*- coding: utf-8 -*-
# @Time         : 2022/5/18 10:26
# @Author       : Yufan Liu
# @Description  : Inference and generation of CDRH3 with trained model


import numpy as np
import torch

amino_vocab = {'B': 1,  # for begin
               'A': 2,
               'C': 3,
               'D': 4,
               'E': 5,
               'F': 6,
               'G': 7,
               'H': 8,
               'I': 9,
               'K': 10,
               'L': 11,
               'M': 12,
               'N': 13,
               'P': 14,
               'Q': 15,
               'R': 16,
               'S': 17,
               'T': 18,
               'V': 19,
               'W': 20,
               'Y': 21,
               'J': 22,  # for end
               }
transfer_vocab = {k: v for v, k in amino_vocab.items()}


class CDRGenerate(object):
    def __init__(self, antigen, model):
        self.model = model
        self.antigen = antigen.unsqueeze(0)

    def generate(self, size, max_length, temperature=None, p=None, method='temperature'):
        assert method in ['temperature', 'top-k', 'top-p', 'greedy'], "Method not supported."
        self.model.eval()

        if method == 'temperature':
            if temperature is not None:
                # sampling with temperature
                sampling = []
                nll_record = []

                for i in range(size):
                    start = [1]
                    out_seq = ''
                    nll = 0

                    while True:
                        start_tensor = torch.tensor(start).unsqueeze(0).long()
                        out, _ = self.model.decoder(start_tensor, self.antigen, None, None)  # [1, 21]
                        probs_without_t = torch.nn.Softmax(-1)(out)[0, -1, :].detach().numpy()
                        probs = torch.nn.Softmax(-1)(out / temperature)[0, -1, :].detach().numpy()
                        for_add = np.random.choice(range(2, 23), p=probs)

                        if len(out_seq) > max_length:
                            break
                        elif transfer_vocab[for_add] == 'J':
                            break
                        nll += -np.log(probs_without_t[for_add - 2])
                        start.append(for_add)
                        out_seq += transfer_vocab[for_add]
                    sampling.append(out_seq)
                    nll_record.append(nll)
                return sampling, nll_record
            else:
                raise KeyError("Temperature value is None with temperature sampling.")

        elif method == 'top-p':
            if p is not None:
                sampling = []
                nll_record = []

                for i in range(size):
                    start = [1]
                    nll = 0
                    out_seq = ''

                    while True:
                        start_tensor = torch.tensor(start).unsqueeze(0).long()
                        out, _ = self.model.decoder(start_tensor, self.antigen, None, None)
                        probs = torch.nn.Softmax(-1)(out)[0, -1, :].detach().numpy()

                        sorted_logits, sorted_indices = torch.sort(out[0, -1, :], descending=True)
                        cumulative_probs = torch.cumsum(torch.nn.Softmax(-1)(sorted_logits), -1)
                        indices_keep = cumulative_probs < 0.9
                        if indices_keep.sum() == 0:
                            indices_keep[0] = True
                        sorted_indices = sorted_indices[indices_keep].detach().numpy()
                        sorted_logits = torch.nn.Softmax(-1)(sorted_logits[:len(sorted_indices)]).detach().numpy()
                        for_add = np.random.choice(sorted_indices, p=sorted_logits) + 2

                        if len(out_seq) > max_length:
                            break
                        elif transfer_vocab[for_add] == 'J':
                            break
                        nll += -np.log(probs[for_add - 2])
                        start.append(for_add)
                        out_seq += transfer_vocab[for_add]

                    sampling.append(out_seq)
                    nll_record.append(nll)
                return sampling, nll_record
            else:
                raise KeyError("Top-p value is None with top-p sampling.")

        else:
            pass


if __name__ == '__main__':
    """
    tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False,
                                              cache_dir='../absolute_antibody/cache_model')
    pretrained_model = BertModel.from_pretrained("Rostlab/prot_bert",
                                                 cache_dir='../absolute_antibody/cache_model')
    antigen_seq = 'KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNSQATNRNTDGSTDYGVLQINSRWWCND' \
                  'GRTPGSRNLCNIPCSALQSSDITATANCAKKIVSDGNGMNAWVAWRKHCKGTDVRVWIKGCRL'
    antigen_seq = ' '.join(list(antigen_seq))
    antigen_encode = pretrained_model(**tokenizer(antigen_seq,
                                                  return_tensors='pt')).last_hidden_state.squeeze(0)[1:-1, :].detach()
    s = time.time()
    generator = CDRGenerate(setting='./setting.yaml',
                            checkpoint='experiment_outs/epoch=424-step=40800.ckpt',
                            antigen=antigen_encode)
    print(generator.generate(100, 40, p=0.9, method='top-p'))
    print(f"Running time: {time.time() - s}s.")
    print("Remote test here")
    """
    pass
