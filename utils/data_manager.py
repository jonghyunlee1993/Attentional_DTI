import torch
from torch.utils.data import DataLoader, Dataset
from tdc.multi_pred import DTI
from transformers import BertTokenizer
from functools import partial

def load_dti_dataset(name="davis"):
    if name not in ["davis", "kiba"]:
        print("choose DTI dataset between kiba or davis")
        raise
    
    dataset = DTI(name=name)
    if name == "davis":
        dataset.convert_to_log(form='binding')    
    dataset_split = dataset.get_split()

    train_df = dataset_split["train"]
    valid_df = dataset_split["valid"]
    test_df = dataset_split["test"]

    return train_df, valid_df, test_df


class DTIDataset(Dataset):
    def __init__(self, data, molecule_tokenizer, protein_tokenizer, molecule_max_len=100, protein_max_len=512):
        self.data = data
        
        self.molecule_max_len = molecule_max_len
        self.protein_max_len = protein_max_len
        
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


# def collate_batch(batch):
#     molecule_seq, protein_seq, y = [], [], []
    
#     for (molecule_seq_, protein_seq_, y_) in batch:
#         molecule_seq.append(molecule_seq_)
#         protein_seq.append(protein_seq_)
#         y.append(y_)
        
#     molecule_seq = molecule_tokenizer.pad(molecule_seq, return_tensors="pt")
#     protein_seq = protein_tokenizer.pad(protein_seq, return_tensors="pt")
#     y = torch.tensor(y).float()
    
#     return molecule_seq, protein_seq, y

class CollateBatch(object):
    def __init__(self, molecule_tokenizer, protein_tokenizer):
        self.molecule_tokenizer = molecule_tokenizer
        self.protein_tokenizer = protein_tokenizer
    
    def __call__(self, batch):
        molecule_seq, protein_seq, y = [], [], []

        for (molecule_seq_, protein_seq_, y_) in batch:
            molecule_seq.append(molecule_seq_)
            protein_seq.append(protein_seq_)
            y.append(y_)

        molecule_seq = self.molecule_tokenizer.pad(molecule_seq, return_tensors="pt")
        protein_seq = self.protein_tokenizer.pad(protein_seq, return_tensors="pt")
        y = torch.tensor(y).float()

        return molecule_seq, protein_seq, y


def custom_dataloader(train_df, valid_df, test_df, batch_size, molecule_max_len=100, protein_max_len=512):

    molecule_tokenizer = BertTokenizer.from_pretrained("data/drug/molecule_tokenizer", model_max_length=128)
    protein_tokenizer = BertTokenizer.from_pretrained("data/target/protein_tokenizer", do_lower_case=False)

    

    train_dataset = DTIDataset(train_df, molecule_tokenizer, protein_tokenizer, molecule_max_len, protein_max_len)
    valid_dataset = DTIDataset(valid_df, molecule_tokenizer, protein_tokenizer, molecule_max_len, protein_max_len)
    test_dataset = DTIDataset(test_df, molecule_tokenizer, protein_tokenizer, molecule_max_len, protein_max_len)

    collate_batch = CollateBatch(molecule_tokenizer, protein_tokenizer)

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=12, 
                                pin_memory=True, prefetch_factor=10, 
                                collate_fn=collate_batch, 
                                drop_last=True, shuffle=True)

    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=12, 
                                pin_memory=True, prefetch_factor=10, 
                                collate_fn=collate_batch)

    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=12, 
                                pin_memory=True, prefetch_factor=10, 
                                collate_fn=collate_batch)

    return train_data_loader, valid_data_loader, test_data_loader