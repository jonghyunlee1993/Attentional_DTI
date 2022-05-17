import torch
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics.functional import mean_squared_error, mean_absolute_error


class DTI_prediction(pl.LightningModule):
    def __init__(self, attentional_dti, **kwars):
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

    
    def configure_optimizers(self, learning_rate=1e-4, t_max=10):
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
        
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def define_callbacks(project_name="attentional_dti"):    
    dirpath = "weights/Attentional_DIT_" + project_name

    callbacks = [
        ModelCheckpoint(monitor='valid_loss', save_top_k=5, dirpath=dirpath, filename='attentional_dti-{epoch:03d}-{valid_loss:.4f}-{valid_mae:.4f}'),
    ]

    return callbacks

def load_checkpoint(model, fpath):
    model = model.load_from_checkpoint(fpath)

    return model