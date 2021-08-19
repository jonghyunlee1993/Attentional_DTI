import torch
from tqdm import tqdm


def train(model, iterator, optimizer, criterion, device, clip=1):
    model.train()

    epoch_loss = 0
    epoch_corrects = 0
    epoch_num_data = 0

    for batch, (X, target, segment_emb, masking_label) in enumerate(tqdm(iterator)):
        
        optimizer.zero_grad()
        
        output = model(X.to(device), segment_emb.long().to(device))
        output_dim = output.shape[-1]
        
#         output = output[masking_label.bool().to(device)].reshape(-1, output_dim)
#         target = target.reshape(-1)[masking_label.reshape(-1).bool()].to(device)
        output = output.reshape(-1, output_dim)
        target = target.reshape(-1).to(device)
        loss   = criterion(output, target)
        loss.backward()
        
        epoch_loss += loss.item()
                
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        output_ = output.clone().detach().to('cpu')
        target_ = target.clone().detach().to('cpu')

        unpad_mask      = (target_ != 1)
        epoch_corrects += (output_.argmax(1)[unpad_mask] == target_[unpad_mask]).float().sum()
        epoch_num_data += len(unpad_mask)
        
    return epoch_loss / len(iterator), 100 * epoch_corrects / epoch_num_data


@torch.no_grad()
def evaluate(model, iterator, optimizer, criterion, device):
    model.eval()
    
    epoch_loss = 0
    epoch_corrects = 0
    epoch_num_data = 0

    for batch, (X, target, segment_emb, masking_label) in enumerate(iterator):
        optimizer.zero_grad()
        
        output = model(X.to(device), segment_emb.long().to(device))
        output_dim = output.shape[-1]
        
#         output = output[masking_label.bool().to(device)].reshape(-1, output_dim)
#         target = target.reshape(-1)[masking_label.reshape(-1).bool()].to(device)
        output = output.reshape(-1, output_dim)
        target = target.reshape(-1).to(device)

        loss   = criterion(output, target)
        
        epoch_loss += loss.item()
        
        output_ = output.clone().detach().to('cpu')
        target_ = target.clone().detach().to('cpu')
        
        unpad_mask      = (target_ != 1)
        epoch_corrects += (output_.argmax(1)[unpad_mask] == target_[unpad_mask]).float().sum()
        epoch_num_data += len(unpad_mask)
        
    return epoch_loss / len(iterator), 100 * epoch_corrects / epoch_num_data


@torch.no_grad()
def predict(model, iterator, device, tokenizer):
    model.eval()
    
    for batch, (X, target, segment_emb, masking_label) in enumerate(iterator):
        output = model(X.to(device), segment_emb.long().to(device))
    
        output_ = torch.argmax(output.clone().detach().to("cpu"), axis=-1)
        target_ = target.clone().detach().to("cpu")

        output_list = decode(output_, tokenizer)
        target_list = decode(target_, tokenizer)

    return output_list, target_list


def decode(x, tokenizer):
    results = []
    for line in x:
        decoded = ""
        for s in line:
            decoded += tokenizer.vocab.itos[s]
        results.append(decoded)
        
    return results 