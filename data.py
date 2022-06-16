# -*- coding: utf-8 -*-
# @Time         : 2022/5/10 18:05
# @Author       : Yufan Liu
# @Description  : Dataset class for autoregressive antibody generation

import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl


class PairedBinder(Dataset):
    # paired antigen and antibody binder data
    def __init__(self, mode='train', species='murine', file=None):
        super(PairedBinder, self).__init__()
        assert mode in ['train', 'valid', 'test'], 'Error in dataset type.'
        assert species in ['murine', 'human'], 'Murine or human supported.'

        if species == 'murine':
            if mode == 'train':
                self.data = pickle.load(open("dataset/abag_train.pkl", 'rb'))[:100]
            elif mode == 'valid':
                self.data = pickle.load(open('dataset/abag_valid.pkl', 'rb'))[:10]
            else:
                if file is None:
                    raise KeyError("No test file provided")
                else:
                    self.data = pickle.load((open(file, 'rb')))

        elif species == 'human':
            if mode == 'train':
                self.data = pickle.load(open("dataset/human_abag_train.pkl", 'rb'))
            elif mode == 'valid':
                self.data = pickle.load(open('dataset/human_abag_valid.pkl', 'rb'))
            else:
                if file is None:
                    raise KeyError("No test file provided")
                else:
                    self.data = pickle.load((open(file, 'rb')))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


class DataCollator(object):
    def __call__(self, item):
        antigen_list = [it[0] for it in item]
        max_ag_length = max(len(x) for x in antigen_list)
        antigen_list = [np.pad(x, ((0, max_ag_length - len(x)), (0, 0))) for x in antigen_list]
        antigen_list = [torch.from_numpy(x).float() for x in antigen_list]
        antigen_stack = torch.stack(antigen_list)

        antibody_list = [it[1] for it in item]

        ab_input = [np.delete(x, [1, 2, 3])[:-1] for x in antibody_list]  # delete framework aa
        max_ab_input = max(len(x) for x in ab_input)
        ab_input = [np.pad(x, (0, max_ab_input - len(x))) for x in ab_input]
        ab_input = [torch.from_numpy(x).float() for x in ab_input]
        ab_input = torch.stack(ab_input)

        ab_output = [np.delete(x, [1, 2, 3])[1:] for x in antibody_list]
        max_ab_output = max(len(x) for x in ab_output)
        ab_output = [np.pad(x, (0, max_ab_output - len(x))) for x in ab_output]
        ab_output = [torch.from_numpy(x).float() for x in ab_output]
        ab_output = torch.stack(ab_output)

        return antigen_stack, \
               ab_input.long(), \
               ab_output.long()


class AntigenAntibodyPairedData(pl.LightningDataModule):
    def __init__(self, batch_size, workers, species):
        super(AntigenAntibodyPairedData, self).__init__()
        self.collate_fn = DataCollator()
        self.train = PairedBinder(mode='train', species=species)
        self.valid = PairedBinder(mode='valid', species=species)
        self.batch_size = batch_size
        self.workers = workers

    def train_dataloader(self):
        # torch.multiprocessing.set_start_method('spawn')
        return DataLoader(self.train, self.batch_size, collate_fn=self.collate_fn, num_workers=self.workers,
                          pin_memory=False, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid, self.batch_size, collate_fn=self.collate_fn, num_workers=self.workers,
                          pin_memory=False, shuffle=False)


if __name__ == '__main__':
    data_module = DataLoader(PairedBinder(), 4, collate_fn=DataCollator())
    x = next(iter(data_module))
    print('x')
