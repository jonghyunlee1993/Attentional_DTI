import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import einsum
from einops import rearrange
from torch.utils.data import DataLoader, Dataset, RandomSampler

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics.functional import mean_squared_error, mean_absolute_error

from transformers import BertModel, BertTokenizer

import pandas as pd
from tqdm import tqdm
from tdc.multi_pred import DTI

davis = DTI(name="davis")
davis.convert_to_log(form = 'binding')
davis_split = davis.get_split()

train_df = davis_split['train']
valid_df = davis_split['valid']
test_df = davis_split['test']

class DTIDataset(Dataset):
    def __init__(self, data, molecule_tokenizer, protein_tokenizer):
        self.data = data
        
        self.molecule_max_len = 128
        self.protein_max_len = 1024
        
        self.molecule_tokenizer = molecule_tokenizer
        self.protein_tokenizer = protein_tokenizer
    
        
    def molecule_encode(self, molecule_sequence):
        molecule_sequence = self.molecule_tokenizer(
            " ".join(molecule_sequence), 
            max_length=self.molecule_max_len, 
            truncation=True
        )
        
        return molecule_sequence
    
    
    def protein_encode(self, protein_sequence):
        protein_sequence = self.protein_tokenizer(
            " ".join(protein_sequence), 
            max_length=self.protein_max_len, 
            truncation=True
        )
        
        return protein_sequence
        
        
    def __len__(self):
        return len(self.data)

    
    def __getitem__(self, idx):
        molecule_sequence = self.molecule_encode(self.data.loc[idx, "Drug"])
        protein_sequence = self.protein_encode(self.data.loc[idx, "Target"])
        y = self.data.loc[idx, "Y"]
                
        return molecule_sequence, protein_sequence, y


def collate_batch(batch):
    molecule_seq, protein_seq, y = [], [], []
    
    for (molecule_seq_, protein_seq_, y_) in batch:
        molecule_seq.append(molecule_seq_)
        protein_seq.append(protein_seq_)
        y.append(y_)
        
    molecule_seq = molecule_tokenizer.pad(molecule_seq, return_tensors="pt")
    protein_seq = protein_tokenizer.pad(protein_seq, return_tensors="pt")
    y = torch.tensor(y).float()
    
    return molecule_seq, protein_seq, y


molecule_tokenizer = molecule_tokenizer = BertTokenizer.from_pretrained("data/drug/molecule_tokenizer", model_max_length=128)
protein_tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)

train_dataset = DTIDataset(train_df, molecule_tokenizer, protein_tokenizer)
valid_dataset = DTIDataset(valid_df, molecule_tokenizer, protein_tokenizer)
test_dataset = DTIDataset(test_df, molecule_tokenizer, protein_tokenizer)

train_dataloader = DataLoader(train_dataset, batch_size=8, num_workers=32, 
                              pin_memory=True, prefetch_factor=10, 
                              drop_last=True, collate_fn=collate_batch, shuffle=True)

valid_dataloader = DataLoader(valid_dataset, batch_size=8, num_workers=32, 
                              pin_memory=True, prefetch_factor=10, 
                              collate_fn=collate_batch)

test_dataloader = DataLoader(test_dataset, batch_size=8, num_workers=32, 
                             pin_memory=True, prefetch_factor=10, 
                             collate_fn=collate_batch)


class CrossAttention(nn.Module):
    def __init__(self, input_dim=128, intermediate_dim=512, heads=8, dropout=0.1):
        super().__init__()
        project_out = input_dim

        self.heads = heads
        self.scale = (input_dim / heads) ** -0.5

        self.key = nn.Linear(input_dim, intermediate_dim, bias=False)
        self.value = nn.Linear(input_dim, intermediate_dim, bias=False)
        self.query = nn.Linear(input_dim, intermediate_dim, bias=False)

        self.out = nn.Sequential(
            nn.Linear(intermediate_dim, project_out),
            nn.Dropout(dropout)
        )

        
    def forward(self, data):
        b, n, d, h = *data.shape, self.heads

        k = self.key(data)
        k = rearrange(k, 'b n (h d) -> b h n d', h=h)

        v = self.value(data)
        v = rearrange(v, 'b n (h d) -> b h n d', h=h)
        
        # get only cls token
        q = self.query(data[:, 0].unsqueeze(1))
        q = rearrange(q, 'b n (h d) -> b h n d', h=h)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attention = dots.softmax(dim=-1)

        output = einsum('b h i j, b h j d -> b h i d', attention, v)
        output = rearrange(output, 'b h n d -> b n (h d)')
        output = self.out(output)
        
        return output


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
        
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

    
class CrossAttentionLayer(nn.Module):
    def __init__(self, 
                 molecule_dim=128, molecule_intermediate_dim=256,
                 protein_dim=1024, protein_intermediate_dim=2048,
                 cross_attn_depth=1, cross_attn_heads=4, dropout=0.1):
        super().__init__()

        self.cross_attn_layers = nn.ModuleList([])
        
        for _ in range(cross_attn_depth):
            self.cross_attn_layers.append(nn.ModuleList([
                nn.Linear(molecule_dim, protein_dim),
                nn.Linear(protein_dim, molecule_dim),
                PreNorm(protein_dim, CrossAttention(
                    protein_dim, protein_intermediate_dim, cross_attn_heads, dropout
                )),
                nn.Linear(protein_dim, molecule_dim),
                nn.Linear(molecule_dim, protein_dim),
                PreNorm(molecule_dim, CrossAttention(
                    molecule_dim, molecule_intermediate_dim, cross_attn_heads, dropout
                ))
            ]))

            
    def forward(self, molecule, protein):
        for i, (f_sl, g_ls, cross_attn_s, f_ls, g_sl, cross_attn_l) in enumerate(self.cross_attn_layers):
            
            cls_molecule = molecule[:, 0]
            x_molecule = molecule[:, 1:]
            
            cls_protein = protein[:, 0]
            x_protein = protein[:, 1:]

            # Cross attention for protein sequence
            cal_q = f_ls(cls_protein.unsqueeze(1))
            cal_qkv = torch.cat((cal_q, x_molecule), dim=1)
            # add activation function
            cal_out = cal_q + cross_attn_l(cal_qkv)
            cal_out = F.gelu(g_sl(cal_out))
            protein = torch.cat((cal_out, x_protein), dim=1)

            # Cross attention for molecule sequence
            cal_q = f_sl(cls_molecule.unsqueeze(1))
            cal_qkv = torch.cat((cal_q, x_protein), dim=1)
            # add activation function
            cal_out = cal_q + cross_attn_s(cal_qkv)
            cal_out = F.gelu(g_ls(cal_out))
            molecule = torch.cat((cal_out, x_molecule), dim=1)
            
        return molecule, protein
    
    
class AttentionalDTI(nn.Module):
    def __init__(self, 
                 molecule_encoder, protein_encoder, cross_attention_layer, 
                 molecule_input_dim=128, protein_input_dim=1024, hidden_dim=512, **kwargs):
        super().__init__()
        self.molecule_encoder = molecule_encoder
        self.protein_encoder = protein_encoder
        
        # model freezing without last layer
        for param in self.molecule_encoder.encoder.layer[0:-1].parameters():
            param.requires_grad = False        
        for param in self.protein_encoder.encoder.layer[0:-1].parameters():
            param.requires_grad = False
        
        self.cross_attention_layer = cross_attention_layer
        
        self.molecule_mlp = nn.Sequential(
            nn.LayerNorm(molecule_input_dim),
            nn.Linear(molecule_input_dim, hidden_dim)
        )
        
        self.protein_mlp = nn.Sequential(
            nn.LayerNorm(protein_input_dim),
            nn.Linear(protein_input_dim, hidden_dim)
        )
        
        self.fc_out = nn.Linear(hidden_dim, 1)
        
    
    def forward(self, molecule_seq, protein_seq):
        encoded_molecule = self.molecule_encoder(**molecule_seq)
        encoded_protein = self.protein_encoder(**protein_seq)
        
        molecule_out, protein_out = self.cross_attention_layer(encoded_molecule.last_hidden_state, encoded_protein.last_hidden_state)
        
        molecule_out = molecule_out[:, 0]
        protein_out = protein_out[:, 0]
        
        # cls token
        molecule_projected = self.molecule_mlp(molecule_out)
        protein_projected = self.protein_mlp(protein_out)
        
        out = self.fc_out(molecule_projected + protein_projected)
        
        return out

molecule_bert = BertModel.from_pretrained("weights/molecule_bert")
protein_bert = BertModel.from_pretrained("weights/protein_bert")
cross_attention_layer = CrossAttentionLayer()
attentional_dti = AttentionalDTI(molecule_bert, protein_bert, cross_attention_layer, cross_attn_depth=4)

class DTI_prediction(pl.LightningModule):
    def __init__(self, attentional_dti):
        super().__init__()
        self.model = attentional_dti

        
    def forward(self, molecule_sequence, protein_sequence):
        return self.model(molecule_sequence, protein_sequence)
    
    
    def training_step(self, batch, batch_idx):
        molecule_sequence, protein_sequence, y = batch
        
        y_hat = self(molecule_sequence, protein_sequence).squeeze(-1)        
        loss = F.mse_loss(y_hat, y)
        
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_mae", mean_absolute_error(y_hat, y), on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        molecule_sequence, protein_sequence, y = batch
        
        y_hat = self(molecule_sequence, protein_sequence).squeeze(-1)        
        loss = F.mse_loss(y_hat, y)
        
        self.log('valid_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("valid_mae", mean_absolute_error(y_hat, y), on_step=False, on_epoch=True, prog_bar=True)
    
    
    def test_step(self, batch, batch_idx):
        molecule_sequence, protein_sequence, y = batch
        
        y_hat = self(molecule_sequence, protein_sequence).squeeze(-1)        
        loss = F.mse_loss(y_hat, y)
        
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_mae", mean_absolute_error(y_hat, y), on_step=False, on_epoch=True, prog_bar=True)
    
    
    def predict_step(self, batch, batch_idx):
        molecule_sequence, protein_sequence, y = batch
        
        y_hat = self(molecule_sequence, protein_sequence).squeeze(-1)        
        
        return y_hat

    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    
    
callbacks = [
    ModelCheckpoint(monitor='valid_loss', save_top_k=5, dirpath='weights/Attentional_DTI_cross_attention_davis', filename='attentional_dti-{epoch:03d}-{valid_loss:.4f}-{valid_mae:.4f}'),
]

model = DTI_prediction(attentional_dti)
trainer = pl.Trainer(max_epochs=100, tpu_cores=8, enable_progress_bar=True, callbacks=callbacks)

trainer.fit(model, train_dataloader, valid_dataloader)