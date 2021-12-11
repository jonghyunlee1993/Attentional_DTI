import torch
import torch.nn as nn


class BERTEmbedding(nn.Module):
    def __init__(self, vocab_dim, seq_len, d_model, dropout_rate):
        super(BERTEmbedding, self).__init__()
        self.vocab_dim = vocab_dim
        self.seq_len = seq_len
        self.d_model = d_model
        self.dropout_rate = dropout_rate
        
        # vocab --> embedding
        self.token_embedding = nn.Embedding(self.vocab_dim, self.d_model) 
        self.token_dropout = nn.Dropout(self.dropout_rate)    
        
        # seq len --> embedding
        self.positional_embedding = nn.Embedding(self.seq_len, self.d_model)
        self.positional_dropout   = nn.Dropout(self.dropout_rate) 
        
        
    def forward(self, data):
        device = data.get_device()
        
        token_embedding = self.token_embedding(data)
        token_embedding = self.token_dropout(token_embedding)
        
        positional_encoding = torch.arange(start=0, end=self.seq_len, step=1).long()
        positional_encoding = positional_encoding.unsqueeze(0).expand(data.size()).to(device)
#         positional_encoding = positional_encoding.unsqueeze(0).expand(data.size())
        
        positional_embedding = self.positional_embedding(positional_encoding)
        positional_embedding = self.positional_dropout(positional_embedding)
        
        return token_embedding + positional_embedding
    

class BERT(nn.Module):
    def __init__(self, vocab_dim, seq_len, d_model, dim_feedforward, pad_token_id, nhead, num_layers, dropout_rate):
        super(BERT, self).__init__()
        self.pad_token_id = pad_token_id
        self.nhead = nhead
        self.embedding = BERTEmbedding(vocab_dim, seq_len, d_model, dropout_rate)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, dim_feedforward=dim_feedforward, nhead=nhead, batch_first=True)
        self.encoder_block = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        
        
    def forward(self, data):
        pad_mask = BERT.get_attn_pad_mask(data, data, self.pad_token_id).repeat(self.nhead, 1, 1)
        embedding = self.embedding(data)
        output = self.encoder_block(embedding, pad_mask) 
        
        return output
    
    
    @staticmethod
    def get_attn_pad_mask(seq_q, seq_k, i_pad):
        batch_size, len_q = seq_q.size()
        batch_size, len_k = seq_k.size()
        pad_attn_mask = seq_k.data.eq(i_pad)
        pad_attn_mask = pad_attn_mask.unsqueeze(1).expand(batch_size, len_q, len_k)
        
        return pad_attn_mask
    
    
class MLMHead(nn.Module):
    def __init__(self, bert, d_model, output_dim, use_RNN=False):
        super(MLMHead, self).__init__()
        self.bert = bert
        self.use_RNN = use_RNN
        self.fc = nn.Linear(d_model, output_dim)
        
        if self.use_RNN:
            self.rnn  = nn.GRU(d_model, d_model)
        
    
    def forward(self, x):
        output = self.bert(x)

        if self.use_RNN:
            output, hidden = self.rnn(output)

        output = self.fc(output)
        
        return output
