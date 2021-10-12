import torch
from torch.utils.data import Dataset


class MoleculeLangaugeModelDataset(Dataset):
    def __init__(self, data, tokenizer, seq_len=128, masking_rate=0.15):
        super(MoleculeLangaugeModelDataset, self).__init__()

        self.data          = data        
        self.tokenizer     = tokenizer
        self.vocab         = tokenizer.vocab
        self.seq_len       = seq_len
        self.masking_rate  = masking_rate
        
        self.cls_token_id  = self.tokenizer.vocab.stoi[self.tokenizer.init_token]
        self.sep_token_id  = self.tokenizer.vocab.stoi[self.tokenizer.eos_token]
        self.pad_token_id  = self.tokenizer.vocab.stoi[self.tokenizer.pad_token]
        self.mask_token_id = self.tokenizer.vocab.stoi[self.tokenizer.unk_token]
        
    def __getitem__(self, idx):
        try:
            target = self.tokenizer.numericalize(self.data[idx]).squeeze()

            if len(target) < self.seq_len - 2:
                pad_length = self.seq_len - len(target) - 2
            else:
                target = target[:self.seq_len-2]
                pad_length = 0

            masked_sent, masking_label = self.masking(target)

            # MLM
            train = torch.cat([
                torch.tensor([self.cls_token_id]), 
                masked_sent,
                torch.tensor([self.sep_token_id]),
                torch.tensor([self.pad_token_id] * pad_length)
            ]).long().contiguous()

            target = torch.cat([
                torch.tensor([self.cls_token_id]), 
                target,
                torch.tensor([self.sep_token_id]),
                torch.tensor([self.pad_token_id] * pad_length)
            ]).long().contiguous()

            masking_label = torch.cat([
                torch.zeros(1), 
                masking_label,
                torch.zeros(1),
                torch.zeros(pad_length)
            ])

            segment_embedding = torch.zeros(target.size(0))
        
            return train, target, segment_embedding, masking_label
        except:
            return None
        
    
    def __len__(self):
        return len(self.data)
    
    
    def __iter__(self):
        for x in self.data:
            yield x
            
    
    def get_vocab(self):
        return self.vocab

    
    def masking(self, x):
        x             = torch.tensor(x).long().contiguous()
        masking_idx   = torch.randperm(x.size()[0])[:round(x.size()[0] * self.masking_rate) + 1]       
        masking_label = torch.zeros(x.size()[0])
        masking_label[masking_idx] = 1
        x             = x.masked_fill(masking_label.bool(), self.mask_token_id)
        
        return x, masking_label
    
def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def generate_epoch_dataloader(data, seq_len, tokenizer, masking_rate, batch_size, collate_fn, shuffle=True, num_workers=6):
    dataset    = MoleculeLangaugeModelDataset(data=data, seq_len=seq_len, tokenizer=tokenizer, masking_rate=masking_rate)
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
    
    return dataloader


def generate_epoch_prediction_dataloader(data, seq_len, tokenizer, masking_rate, batch_size, collate_fn, shuffle=True, num_workers=5):    
    dataset    = MoleculeLangaugeModelDataset(data=data, seq_len=seq_len, tokenizer=tokenizer, masking_rate=masking_rate)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
    
    return dataloader