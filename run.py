from utils.data_manager import *
from model.attentional_dti import *
from model.trainer import *


if __name__ == "__main__":
    dataset_name = "davis"
    
    molecule_max_len = 100
    protein_max_len = 1000
    
    batch_size = 10
    n_epochs = 300
    t_max = n_epochs / 10
    n_gpus = 2

    train_df, valid_df, test_df = load_dti_dataset(name=dataset_name)
    
    train_data_loader, valid_data_loader, test_data_loader = custom_dataloader(train_df, valid_df, test_df, batch_size, molecule_max_len, protein_max_len)

    molecule_bert, protein_bert = load_encoder()
    cross_attention_layer = CrossAttentionLayer()
    attentional_dti = AttentionalDTI(molecule_bert, protein_bert, cross_attention_layer, cross_attn_depth=4)

    callbacks = define_callbacks(project_name="long_protein_davis")
    model = DTI_prediction(attentional_dti, learning_rate=1e-4, t_max=t_max)

    if n_gpus > 1:
        trainer = pl.Trainer(max_epochs=n_epochs, gpus=n_gpus, enable_progress_bar=True, callbacks=callbacks, strategy="ddp")
    else:   
        trainer = pl.Trainer(max_epochs=n_epochs, gpus=n_gpus, enable_progress_bar=True, callbacks=callbacks)

    trainer.fit(model, train_data_loader, valid_data_loader)