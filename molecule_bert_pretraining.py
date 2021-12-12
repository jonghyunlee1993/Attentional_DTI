import os
import pickle
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from transformers import PreTrainedTokenizerFast
from transformers import BertConfig, BertForMaskedLM
from transformers import DataCollatorForLanguageModeling


class MaskedLMDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        
    def encode(self, data):
        return self.tokenizer.encode(data, max_length=self.max_length, truncation=True)
        
        
    def __len__(self):
        return len(self.data)

    
    def __getitem__(self, idx):
        return torch.tensor(self.encode(self.data[idx]), dtype=torch.long)
    

class Bert(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.model = BertForMaskedLM(config)
        self.train_accuracy = torchmetrics.Accuracy()
        self.valid_accuracy = torchmetrics.Accuracy()
        
        
    def forward(self, input_ids, labels):
        return self.model(input_ids=input_ids, labels=labels)

       
    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        labels = batch['labels']
        
        output = self(input_ids, labels)

        loss = output.loss
        logits = output.logits

        preds = logits.argmax(dim=-1)
        
        self.log('train_loss', float(loss), on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_accuracy", self.train_accuracy(preds[labels > 0], labels[labels > 0]), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    
    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        labels = batch['labels']
        
        output = self(input_ids, labels)

        loss = output.loss
        logits = output.logits

        preds = logits.argmax(dim=-1)
        
        self.log('valid_loss', float(loss), on_step=False, on_epoch=True, prog_bar=True)
        self.log("valid_accuracy", self.valid_accuracy(preds[labels > 0], labels[labels > 0]), on_step=False, on_epoch=True, prog_bar=True, logger=True)

    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "valid_loss"}


if __name__ == "__main__":

    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file="data/drug/tokenizer_model/tokenizer.json",
        pad_token="[PAD]",
        mask_token="[MASK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        unk_token="[UNK]"
    )

    vocab_size = len(fast_tokenizer.get_vocab().keys())

    print(f"load tokenizer\nvocab size: {vocab_size}\nspecial tokens: {fast_tokenizer.all_special_tokens}")

    if not os.path.exists("data/drug/X.pkl"):
        from sklearn.model_selection import train_test_split
        
        with open("data/drug/molecule_total.txt", 'r') as f:
            data = f.readlines()
        
            print(f"load dataset ... # of data: {len(data)}")
        
        X_train, X_test = train_test_split(data, test_size=0.2, random_state=42, shuffle=True)
        X_train, X_valid = train_test_split(X_train, test_size=0.2, random_state=42, shuffle=True)
        
        with open("data/drug/X.pkl", "wb") as f:
            pickle.dump([X_train, X_valid, X_test], f)
    else:
        with open("data/drug/X.pkl", "rb") as f:
            X_train, X_valid, X_test = pickle.load(f)

    print(f"load dataset\nX_train: {len(X_train)}\nX_valid: {len(X_valid)}\nX_test: {len(X_test)}")

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=fast_tokenizer, mlm=True, mlm_probability=0.3
    )

    train_dataset = MaskedLMDataset(X_train, fast_tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=512*8, collate_fn=data_collator, num_workers=4*8)

    valid_dataset = MaskedLMDataset(X_valid, fast_tokenizer)
    valid_loader = DataLoader(valid_dataset, batch_size=512*8, collate_fn=data_collator, num_workers=4*8)

    test_dataset = MaskedLMDataset(X_test, fast_tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=512*8, collate_fn=data_collator, num_workers=4*8)

    config = BertConfig(
        vocab_size=vocab_size,
        hidden_size=128,
        num_hidden_layers=8,
        num_attention_heads=8,
        intermediate_size=512,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=128,
        type_vocab_size=1,
        pad_token_id=0,
        position_embedding_type="absolute"
    )

    model = Bert(config)
    callbacks = [
        ModelCheckpoint(monitor='valid_loss', dirpath='weights/molecule_bert', filename='molecule_bert-{epoch:03d}-{valid_loss:.4f}'),
        EarlyStopping('valid_loss', patience=10)
    ]

    trainer = pl.Trainer(max_epochs=100, tpu_cores=8, enable_progress_bar=True, callbacks=callbacks)
    trainer.fit(model, test_loader, test_loader)