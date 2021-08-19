import torch
import torch.nn as nn


class BERT(nn.Module):
    def __init__(self, vocab_dim, seq_len, embedding_dim, pad_token_id, device):
        super(BERT, self).__init__()
        print(device)
        self.pad_token_id  = pad_token_id
        self.nhead         = 8
        self.embedding     = BERTEmbedding(vocab_dim, seq_len, embedding_dim, device, dropout_rate=0.1)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=self.nhead, batch_first=True)
        self.encoder_block = nn.TransformerEncoder(self.encoder_layer, num_layers=8)
        
        
    def forward(self, data, segment_embedding):
        pad_mask  = BERT.get_attn_pad_mask(data, data, self.pad_token_id).repeat(self.nhead, 1, 1)
        embedding = self.embedding(data, segment_embedding)
        output    = self.encoder_block(embedding, pad_mask) 
        
        return output
    
    @staticmethod
    def get_attn_pad_mask(seq_q, seq_k, i_pad):
        batch_size, len_q = seq_q.size()
        batch_size, len_k = seq_k.size()
        pad_attn_mask = seq_k.data.eq(i_pad)
        pad_attn_mask= pad_attn_mask.unsqueeze(1).expand(batch_size, len_q, len_k)
        
        return pad_attn_mask


class BERTEmbedding(nn.Module):
    def __init__(self, vocab_dim, seq_len, embedding_dim, device, dropout_rate=0.1):
        super(BERTEmbedding, self).__init__()
        self.seq_len       = seq_len
        self.vocab_dim     = vocab_dim
        self.embedding_dim = embedding_dim
        self.dropout_rate  = dropout_rate
        
        # vocab --> embedding
        self.token_embedding      = nn.Embedding(self.vocab_dim, self.embedding_dim) 
        self.token_dropout        = nn.Dropout(self.dropout_rate)    
        
        # seq len --> embedding
        self.positional_embedding = nn.Embedding(self.seq_len, self.embedding_dim)
        self.positional_dropout   = nn.Dropout(self.dropout_rate) 
        
        # segment (0, 1) --> embedding
        self.segment_embedding    = nn.Embedding(2, self.embedding_dim)
        self.segment_dropout      = nn.Dropout(self.dropout_rate) 
        
        
    def forward(self, data, segment_embedding):
        token_embedding      = self.token_embedding(data)
        token_embedding      = self.token_dropout(token_embedding)
        
        positional_encoding  = torch.arange(start=0, end=self.seq_len, step=1).long()
        # data의 device 정보 가져와서 처리
        positional_encoding  = positional_encoding.unsqueeze(0).expand(data.size()).to(device)
        positional_embedding = self.positional_embedding(positional_encoding)
        positional_embedding = self.positional_dropout(positional_embedding)
        
        segment_embedding    = self.segment_embedding(segment_embedding)
        segment_embedding    = self.segment_dropout(segment_embedding)
        
        return token_embedding + positional_embedding + segment_embedding


class MaskedLanguageModeling(nn.Module):
    def __init__(self, bert, output_dim):
        super(MaskedLanguageModeling, self).__init__()
        self.bert = bert
        d_model   = bert.embedding.token_embedding.weight.size(1)
        self.rnn  = nn.GRU(d_model, d_model)
        self.fc   = nn.Linear(d_model, output_dim)
    
    def forward(self, x, segment_embedding):
        output = self.bert(x, segment_embedding)
        output, hidden = self.rnn(output)
        output = self.fc(output)
        
        return output