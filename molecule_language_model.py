import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

import torchtext

import os
import glob
import pickle
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

from model.bert import BERT, MaskedLanguageModeling
from utils.molecule_dataloader import *
from utils.trainer import train, evaluate, predict


def load_dataset():
    print("load dataset ... ")
    with open("data/molecule_net/MoleculeNet_train.pickle", 'rb') as f:
        train_data = pickle.load(f)

    test_data  = train_data[:int(len(train_data) * 0.2)]
    train_data = train_data[int(len(train_data) * 0.2):]
    
    return train_data, test_data


def load_tokenizer():
    print("load tokenizer ... ")
    with open("data/molecule_net/MoleculeNet_tokenizer.pickle", "rb") as f:
        tokenizer = pickle.load(f)

    return tokenizer


def define_model(vocab_dim, seq_len, embedding_dim, device, num_head=4, num_layer=4):
    bert_base = BERT(vocab_dim=vocab_dim, seq_len=seq_len, embedding_dim=embedding_dim, pad_token_id=1, num_head=num_head, num_layer=num_layer).to(device)
    model     = MaskedLanguageModeling(bert_base, output_dim=vocab_dim, use_RNN=True).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, betas=[0.9, 0.999], weight_decay=0.01)
    # scheduler = CosineAnnealingLR(optimizer, T_max=10)
    scheduler = ReduceLROnPlateau(optimizer)
    criterion = nn.CrossEntropyLoss(ignore_index=1)

    return model, optimizer, scheduler, criterion


def check_trained_weights(output_path="output/MoleculeNet/*.tsv", trained_weight='weights/MoleculeNet_LM_best.pt'):
    start_epoch = 0
    if len(glob.glob(output_path)) != 0:
        print(f"load pretrained model : {trained_weight}")
        start_epoch = len(glob.glob(output_path))
        model.load_state_dict(torch.load(trained_weight))

    return start_epoch, model


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings(action='ignore')

    train_data, test_data = load_dataset()
    tokenizer = load_tokenizer()
    
    VOCAB_DIM     = len(tokenizer.vocab.itos)
    SEQ_LEN       = 100
    EMBEDDING_DIM = 128
    DEVICE        = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE    = 2048
    N_EPOCHS      = 1000
    PAITIENCE     = 10

    output_path = "output/MoleculeNet"
    weight_path = "weights"


    model, optimizer, scheduler, criterion = define_model(vocab_dim=VOCAB_DIM, seq_len=SEQ_LEN, embedding_dim=EMBEDDING_DIM, device=DEVICE)

    n_paitience = 0
    best_valid_loss = float('inf')
    optimizer.zero_grad()
    optimizer.step()

    start_epoch, model = check_trained_weights()

    for epoch in range(start_epoch, N_EPOCHS):
#     epoch_masking_rate = np.random.choice([0.3, 0.4, 0.5, 0.6])
        epoch_masking_rate = 0.3
        epoch_train_data   = shuffle(train_data, n_samples=5000000)
        epoch_valid_data   = shuffle(test_data, n_samples=100000)
        train_dataloader   = generate_epoch_dataloader(
                                                        epoch_train_data, 
                                                        seq_len=SEQ_LEN, 
                                                        tokenizer=tokenizer, 
                                                        batch_size=BATCH_SIZE, 
                                                        masking_rate=epoch_masking_rate,
                                                        collate_fn=collate_fn,
                                                        num_workers=10
                                                        )
        
        valid_dataloader   = generate_epoch_dataloader(
                                                        epoch_valid_data, 
                                                        seq_len=SEQ_LEN, 
                                                        tokenizer=tokenizer, 
                                                        batch_size=BATCH_SIZE, 
                                                        masking_rate=0.3,
                                                        collate_fn=collate_fn,
                                                        num_workers=10
                                                        )
        
        print(f'Epoch: {epoch:04} Masking rate: {epoch_masking_rate} Train dataset: {len(epoch_train_data)} Valid dataset: {len(epoch_valid_data)}')
        
        train_loss, train_accuracy = train(model, train_dataloader, optimizer, criterion, DEVICE)
        valid_loss, valid_accuracy = evaluate(model, valid_dataloader, optimizer, criterion, DEVICE)
        
        scheduler.step(valid_loss)
        
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.4f}\nValid Loss: {valid_loss:.4f} | Valid Acc: {valid_accuracy:.4f}')

        with open(os.path.join(output_path, "log.txt"), "a") as f:
            f.write("epoch: {0:04d} train loss: {1:.4f}, train acc: {2:.4f}, test loss: {3:.4f}, test acc: {4:.4f}\n".format(epoch, train_loss, train_accuracy, valid_loss, valid_accuracy))

        if epoch % 1 == 0:
            samples_for_prediction = shuffle(epoch_valid_data, n_samples=100)
            prediction_dataloader  = generate_epoch_prediction_dataloader(
                                                                            samples_for_prediction, 
                                                                            seq_len=SEQ_LEN, 
                                                                            tokenizer=tokenizer, 
                                                                            batch_size=len(samples_for_prediction), 
                                                                            masking_rate=0.3, 
                                                                            collate_fn=collate_fn
                                                                            )
            output_list, target_list = predict(model, prediction_dataloader, DEVICE, tokenizer)
            prediction_results = pd.DataFrame({"output": output_list, "target": target_list})
            prediction_results.to_csv(os.path.join(output_path, "prediction_results_epoch-{0:04d}.tsv".format(epoch)), sep="\t", index=False)            
            
        if n_paitience < PAITIENCE:
            if best_valid_loss > valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), os.path.join(weight_path, 'MoleculeNet_LM_best.pt'))
                n_paitience = 0
            elif best_valid_loss <= valid_loss:
                n_paitience += 1
        else:
            print("Early stop!")
            model.load_state_dict(torch.load(os.path.join(weight_path, 'MoleculeNet_LM_best.pt')))
            model.eval()
            break



