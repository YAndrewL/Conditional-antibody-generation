# -*- coding: utf-8 -*-
# @Time         : 2022/5/11 18:20
# @Author       : Yufan Liu
# @Description  : Vanilla Transformer model
import math
from typing import Dict
from sklearn import metrics

import torch
import torch.nn as nn
from torch import optim
import pytorch_lightning as pl
import numpy as np


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size  # input dim or model dim
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads == embed_size), "Embed_size needs to be divided by heads"

        # K, Q, V
        self.key = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.query = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.value = nn.Linear(self.head_dim, self.head_dim, bias=False)

        # fc
        self.fc_out = nn.Linear(self.head_dim * heads, embed_size, bias=False)  # actually the same

    def forward(self, values, keys, query, mask):
        N = query.shape[0]  # batch size or number of samples

        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]  # input sequence length

        # split input into multi-heads shape -> [Batch, Seq_len, heads, head_dim]
        values = self.value(values.reshape(N, value_len, self.heads, self.head_dim))
        keys = self.key(keys.reshape(N, key_len, self.heads, self.head_dim))
        query = self.query(query.reshape(N, query_len, self.heads, self.head_dim))

        # [batch, length_q, head, head_dim] * [batch, length_k, head, head_dim] -> [batch, head, length_q, length_k]
        energy = torch.einsum('bqhd,bkhd->bhqk', [query, keys])

        # add mask information
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e4"))

        # head_dim or embed_size
        attention = torch.softmax(energy / (self.head_dim ** 0.5), dim=-1)  # apply softmax in last dim

        # [batch, head, length_q, length_k] * [batch, length_v, head, head_dim] -> [batch, length_q, embed_size]
        out = torch.einsum('bhql,blhd->bqhd', [attention, values]).reshape(N, query_len, self.heads * self.head_dim)

        return self.fc_out(out), attention


class SinCosPositionalEncoding(nn.Module):
    # No need for backpropagation
    def __init__(self, device, embed_size, max_len):
        super(SinCosPositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)
        self.device = device

    def forward(self, x):
        """
        x: [batch_size, seq_len, embed_size]
        """
        pos = self.pe[:, :x.size(1), :].to(self.device).expand(x.size(0), -1, -1)
        return pos


class EncoderBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(EncoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, embed_size * forward_expansion),
            nn.ReLU(),
            nn.Linear(embed_size * forward_expansion, embed_size)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, values, keys, query, mask):
        attention, cross_attention = self.attention(values, keys, query, mask)
        x = self.dropout(self.norm1(attention) + query)  # pre-norm
        ff = self.feed_forward(x)
        out = self.dropout(self.norm2(ff) + x)

        return out, cross_attention


# DecoderBlock = SelfAttention + EncoderBlock
class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)  # similar blocks
        self.norm = nn.LayerNorm(embed_size)
        self.encoder_block = EncoderBlock(embed_size, heads, dropout, forward_expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        attention, _ = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention) + x)
        out, cross_attention = self.encoder_block(value, key, query, src_mask)
        return out, cross_attention


# add target input sentence
class Decoder(nn.Module):
    def __init__(
            self,
            ab_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            max_length,
            device,
    ):
        super(Decoder, self).__init__()
        self.aa_embedding = nn.Embedding(ab_vocab_size + 1, embed_size)
        self.position_embedding = SinCosPositionalEncoding(device, embed_size, max_length)

        self.layers = nn.ModuleList(
            [DecoderBlock(embed_size, heads, forward_expansion, dropout) for _ in range(num_layers)])

        self.fc_out = nn.Linear(embed_size, ab_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length)  # expand to batch numbers
        x = self.dropout(self.aa_embedding(x)) + self.dropout(self.position_embedding(positions))

        for layer in self.layers:
            x, cross_attention = layer(x, enc_out, enc_out, src_mask, trg_mask)
        return self.fc_out(x), cross_attention


# consine learning rate warm-up
class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor


class Transformer(pl.LightningModule):
    def __init__(
            self,
            ab_vocab_size=21,
            embed_size=1024,
            num_layers=4,
            forward_expansion=2,
            heads=4,
            dropout=0.5,
            device='cuda',
            max_ab_length=64,
            lr=0.001,
            max_epochs=30,
            warm_up=1,
            experiment_tracking: Dict = None
    ):
        super(Transformer, self).__init__()
        self.save_hyperparameters()  # store in self.hparams
        del self.hparams['experiment_tracking']
        self.decoder = Decoder(ab_vocab_size,
                               embed_size,
                               num_layers,
                               heads,
                               forward_expansion,
                               dropout,
                               max_ab_length,
                               device
                               )
        self.dev = device
        self.lr = lr
        self.max_epochs = max_epochs
        self.warm_up = warm_up
        self.exp_tracking = experiment_tracking

    def on_train_start(self) -> None:
        # experimental tracking
        if self.exp_tracking is not None:
            self.logger.log_hyperparams({**self.exp_tracking, **self.hparams}, {'hp/train_loss': 0, 'hp/val_loss': 0})
        else:
            self.logger.log_hyperparams(self.hparams, {'hp/train_loss': 0, 'hp/val_loss': 0})

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        # broadcast to every head: [Batch, 1, len, len]
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(N, 1, trg_len, trg_len).long()
        return trg_mask.to(self.dev)

    def make_src_mask(self, src):
        src_mask = src.ne(0).int()[:, :, 0].unsqueeze(1).unsqueeze(2)  # [Batch, 1, 1, len_k]
        return src_mask.to(self.dev)

    def forward(self, batch_data):
        ag, ab_input, _ = batch_data
        ab_mask = self.make_trg_mask(ab_input)
        ag_mask = self.make_src_mask(ag)
        out = self.decoder(ab_input, ag, ag_mask, ab_mask)
        return out

    def training_step(self, batch, batch_idx):
        pred_out, cross_attention = self(batch)
        true_target = batch[2]
        criterion = nn.CrossEntropyLoss(ignore_index=-2)
        loss = criterion(pred_out.permute(0, 2, 1), true_target - 2)
        self.log('train_loss', loss, on_step=True, logger=True)
        self.log('hp/train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        pred_out, cross_attention = self(batch)
        true_target = batch[2]
        criterion = nn.CrossEntropyLoss(ignore_index=-2)
        loss = criterion(pred_out.permute(0, 2, 1), true_target - 2)
        self.log('val_loss', loss, on_step=False, logger=True)
        self.log('hp/val_loss', loss)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.lr_scheduler = CosineWarmupScheduler(optimizer, warmup=self.warm_up, max_iters=self.max_epochs)
        return optimizer

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.lr_scheduler.step()
