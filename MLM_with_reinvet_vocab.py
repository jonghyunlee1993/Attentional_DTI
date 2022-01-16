import os 
import re
import copy
import glob
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import torchmetrics
import pytorch_lightning as pl
from transformers import BertConfig, BertForMaskedLM
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


class Vocabulary:
    """Stores the tokens and their conversion to vocabulary indexes."""

    def __init__(self, tokens=None, starting_id=0):
        self._tokens = {}
        self._current_id = starting_id

        if tokens:
            for token, idx in tokens.items():
                self._add(token, idx)
                self._current_id = max(self._current_id, idx + 1)

    def __getitem__(self, token_or_id):
        return self._tokens[token_or_id]

    def add(self, token):
        """Adds a token."""
        if not isinstance(token, str):
            raise TypeError("Token is not a string")
        if token in self:
            return self[token]
        self._add(token, self._current_id)
        self._current_id += 1
        return self._current_id - 1

    def update(self, tokens):
        """Adds many tokens."""
        return [self.add(token) for token in tokens]

    def __delitem__(self, token_or_id):
        other_val = self._tokens[token_or_id]
        del self._tokens[other_val]
        del self._tokens[token_or_id]

    def __contains__(self, token_or_id):
        return token_or_id in self._tokens

    def __eq__(self, other_vocabulary):
        return self._tokens == other_vocabulary._tokens  # pylint: disable=W0212

    def __len__(self):
        return len(self._tokens) // 2

    def encode(self, tokens, max_length=128, masking_rate=0.2):
        """Encodes a list of tokens as vocabulary indexes."""
        vocab_index = np.repeat(0, max_length) # initialize with PAD token
        vocab_index[0] = 1 # CLS token
        
        for i, token in enumerate(tokens):   
            if i <= max_length - 2:
                try:
                    vocab_index[i + 1] = self._tokens[token]
                except:
                    vocab_index[i + 1] = 3 # UNK token
            elif i > max_lenght - 2:
                break
        
        vocab_index[i + 1] = 2 # SEP token    
        
        masked_index = copy.deepcopy(vocab_index)
        mask = np.random.permutation(list(range(1, 1 + min(len(tokens), max_length))))[:round(max_length*masking_rate)]
        masked_index[mask] = 4 # MASK token
        
        return masked_index, vocab_index

    
    def decode(self, vocab_index):
        """Decodes a vocabulary index matrix to a list of tokens."""
        tokens = []
        for idx in vocab_index:
            tokens.append(self[idx])
        return tokens

    def _add(self, token, idx):
        if idx not in self._tokens:
            self._tokens[token] = idx
            self._tokens[idx] = token
        else:
            raise ValueError("IDX already present in vocabulary")

    def tokens(self):
        """Returns the tokens from the vocabulary"""
        return [t for t in self._tokens if isinstance(t, str)]
    
    

class SMILESTokenizer:
    """Deals with the tokenization and untokenization of SMILES."""

    REGEXPS = {
        "brackets": re.compile(r"(\[[^\]]*\])"),
        "2_ring_nums": re.compile(r"(%\d{2})"),
        "brcl": re.compile(r"(Br|Cl)")
    }
    REGEXP_ORDER = ["brackets", "2_ring_nums", "brcl"]

    def tokenize(self, data):
        """Tokenizes a SMILES string."""
        def split_by(data, regexps):
            if not regexps:
                return list(data)
            regexp = self.REGEXPS[regexps[0]]
            splitted = regexp.split(data)
            tokens = []
            for i, split in enumerate(splitted):
                if i % 2 == 0:
                    tokens += split_by(split, regexps[1:])
                else:
                    tokens.append(split)
            return tokens

        tokens = split_by(data, self.REGEXP_ORDER)

        return tokens

    def untokenize(self, tokens):
        """Untokenizes a SMILES string."""
        smi = ""
        for token in tokens:
            if token == "[SEP]":
                break
            if token != "[CLS]":
                smi += token
        return smi


def create_vocabulary(smiles_list, tokenizer):
    """Creates a vocabulary for the SMILES syntax."""
    tokens = set()
    for smi in smiles_list:
        tokens.update(tokenizer.tokenize(smi))

    vocabulary = Vocabulary()
    vocabulary.update(['[PAD]', '[CLS]', '[SEP]', '[UNK]', '[MASK]'] + sorted(tokens))  # end token is 0 (also counts as padding)
    return vocabulary


class MaskedLMDataset(Dataset):
    def __init__(self, data, vocabulary, max_length=128, masking_rate=0.2):
        self.data = data
        self.vocabulary = vocabulary
        self.max_length = max_length
        self.masking_rate = masking_rate
        
        
    def encode(self, data):
        return self.vocabulary.encode(data, max_length=self.max_length, masking_rate=self.masking_rate)
        
        
    def __len__(self):
        return len(self.data)

    
    def __getitem__(self, idx):
        X, y = self.encode(self.data[idx])
        
        return torch.tensor(X, dtype=torch.long), torch.tensor(y, dtype=torch.long)


class Bert(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.model = BertForMaskedLM(config)
        self.train_accuracy = torchmetrics.Accuracy()
        self.valid_accuracy = torchmetrics.Accuracy()
        
        
    def forward(self, X, y):
        return self.model(input_ids=X, labels=y)

       
    def training_step(self, batch, batch_idx):
        X, y = batch
        output = self(X, y)

        loss = output.loss
        logits = output.logits

        preds = logits.argmax(dim=-1)
        
        self.log('train_loss', float(loss), on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_accuracy", self.train_accuracy(preds[y > 0], y[y > 0]), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    
    def validation_step(self, batch, batch_idx):
        X, y = batch
        output = self(X, y)

        loss = output.loss
        logits = output.logits

        preds = logits.argmax(dim=-1)
        
        self.log('valid_loss', float(loss), on_step=False, on_epoch=True, prog_bar=True)
        self.log("valid_accuracy", self.valid_accuracy(preds[y > 0], y[y > 0]), on_step=False, on_epoch=True, prog_bar=True, logger=True)

    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=5e-4)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

        # return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "valid_loss"}
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


if __name__ == "__main__":
    vocab_path = "data/drug/vocabulary.pkl"

    if not os.path.exists(vocab_path):
        from reinvent_chemistry.file_reader import FileReader

        reader = FileReader([], None)
        smiles_list = reader.read_delimited_file("data/drug/molecule_qed_filtered.txt")

        tokenizer = SMILESTokenizer()
        vocabulary = create_vocabulary(smiles_list, tokenizer=tokenizer)
        
        with open(vocab_path, "wb") as f:
            pickle.dump(vocabulary, f)    

    elif os.path.exists(vocab_path):
        with open(vocab_path, "rb") as f:
            vocabulary = pickle.load(f)
    
    vocab_size = len(vocabulary.tokens())
    
    with open("data/drug/molecule_qed_filtered.txt", "r") as f:
        data = f.readlines()
    data = [d.strip() for d in data]  

    X_train, X_test = train_test_split(data, test_size=0.1, random_state=42, shuffle=True)
    X_train, X_valid = train_test_split(X_train, test_size=0.1, random_state=42, shuffle=True)

    print(f"Train: {len(X_train)} Valid: {len(X_valid)} Test: {len(X_test)}")
    
    batch_size = 1024
    num_workers = 16

    train_dataset = MaskedLMDataset(X_train, vocabulary, max_length=128, masking_rate=0.2)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)

    valid_dataset = MaskedLMDataset(X_valid, vocabulary, max_length=128, masking_rate=0.2)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers)

    test_dataset = MaskedLMDataset(X_test, vocabulary, max_length=128, masking_rate=0.2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

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

    # 실험 버전
    exp_v_num = 4
    if exp_v_num:
        dir_path = os.path.join('weights/molecule_bert', "exp" + str(exp_v_num))

        if glob.glob(dir_path + "/*.ckpt"):
            print(f"exp{str(exp_v_num)} is already exists!")
            
            class CustomError(Exception):
                pass
            
            raise CustomError("Change exp_v_num")
    else:
        dir_path = 'weights/molecule_bert'

    model = Bert(config)
    callbacks = [
        ModelCheckpoint(monitor='valid_loss', dirpath=dir_path, filename='molecule_bert-{epoch:03d}-{valid_loss:.4f}'),
        EarlyStopping('valid_loss', patience=3)
    ]

    trainer = pl.Trainer(max_epochs=10, gpus=2, enable_progress_bar=True, callbacks=callbacks, accelerator="ddp", precision=16)
    trainer.fit(model, train_loader, valid_loader)