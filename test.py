import torch
import torch.nn as nn
import torch.optim as optim

import pytorch_lightning as pl
import torchmetrics
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

import os
import glob
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from model.bert import BERT, MLMHead
from utils.molecule_dataloader import MoleculeLangaugeModelDataset, collate_fn
from utils.trainer import train, evaluate, predict


def load_dataset():
    print("load dataset ... ")
#     with open("data/molecule_net/molecule_total.pickle", 'rb') as f:
#         train_data = pickle.load(f)
        
#     train_data = train_data[:1000000]
    with open("data/molecule_net/molecule_small.pickle", "rb") as f:
        train_data = pickle.load(f)
    
    
    train_data, test_data = train_test_split(train_data, test_size=0.2, shuffle=True, random_state=1234)
    train_data, valid_data = train_test_split(train_data, test_size=0.2, shuffle=True, random_state=1234)
    
    return train_data, valid_data, test_data


def load_tokenizer():
    print("load tokenizer ... ")
    with open("data/molecule_net/molecule_tokenizer.pickle", "rb") as f:
        tokenizer = pickle.load(f)

    return tokenizer


train_data, valid_data, test_data = load_dataset()
tokenizer = load_tokenizer()

seq_len = 100
d_model = 128
dim_feedforward = 512
dropout_rate = 0.1
pad_token_id = 3
nhead = 8
num_layers = 8
use_RNN = False
batch_size = 512 * 4
masking_rate = 0.3
vocab_dim = len(tokenizer[0])
learning_rate = 0.0001

train_dataset = MoleculeLangaugeModelDataset(data=train_data, seq_len=seq_len, tokenizer=tokenizer, masking_rate=masking_rate)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, collate_fn=collate_fn, pin_memory=True)

valid_dataset = MoleculeLangaugeModelDataset(data=valid_data, seq_len=seq_len, tokenizer=tokenizer, masking_rate=masking_rate)
valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=8, collate_fn=collate_fn, pin_memory=True)

test_dataset = MoleculeLangaugeModelDataset(data=test_data, seq_len=seq_len, tokenizer=tokenizer, masking_rate=masking_rate)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, collate_fn=collate_fn, pin_memory=True)

DEVICE = "cuda"

bert_base = BERT(vocab_dim, seq_len, d_model, dim_feedforward, pad_token_id, nhead, num_layers, dropout_rate)
model = MLMHead(bert_base, d_model, vocab_dim, use_RNN)

class MoleculeNet(pl.LightningModule):
    def __init__(self):
        super(MoleculeNet, self).__init__()
        bert_base = BERT(vocab_dim, seq_len, d_model, dim_feedforward, pad_token_id, nhead, num_layers, dropout_rate)
        self.model = MLMHead(bert_base, d_model, vocab_dim, use_RNN)
    
    def training_step(self, batch, batch_idx):
        X, y, masking_label = batch

        y_hat = self.model(X)
        loss = F.cross_entropy(y_hat, y, ignore_index=3)
        
        self.log("train_loss", loss)

        return loss

      
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    
molecule_net = MoleculeNet()

trainer = pl.Trainer(tpu_core=8)
trainer.fit(molecule_net, train_dataloader)