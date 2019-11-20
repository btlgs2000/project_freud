import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl


class PerFeatureModel(pl.LightningModule):

    def __init__(self, layers_n, activation, last_activation):
        pass

    def forward(self, x):
        pass

    def training_step(self, batch, batch_nb):
        pass

    def validation_step(self, batch, batch_nb):
        pass

    def validation_end(self, outputs):
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

    @pl.data_loader
    def train_dataloader(self):
        pass

    @pl.data_loader
    def val_dataloader(self):
        pass

    @pl.data_loader
    def test_dataloader(self):
        pass
