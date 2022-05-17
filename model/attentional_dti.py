import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import einsum
from einops import rearrange

from transformers import BertModel


def load_encoder(moleucule_encoder_fpath="weights/molecule_bert", protein_encoder_fpath="weights/protein_bert"):
    molecule_bert = BertModel.from_pretrained(moleucule_encoder_fpath)
    protein_bert = BertModel.from_pretrained(protein_encoder_fpath)

    return molecule_bert, protein_bert
    

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