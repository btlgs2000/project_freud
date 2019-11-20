import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from dataset import DataSet, Record, TupleDataset, StandardNormalizer
from pytorch_utils import TorchTupleDataset
from pytorch_models import SingleFeatureModel as PyTorchSingleFeatureModel, MlpNet


class SingleFeatureModel(pl.LightningModule):

    def __init__(self, layers_n, activation, last_activation):
        self.linearModel = MlpNet([6, 20, 20, 20, 1], F.leaky_relu, torch.sigmoid)
        self.model = PyTorchSingleFeatureModel(self.linearModel)

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


if __name__ == '__main__':
    # carico i dataset
    TRAINING_SET_PATH = r"C:\Users\btlgs\Documents\Project_fraud_2\sestuple_datasets\training_set_classic_sestuple"
    DEV_SET_PATH = r"C:\Users\btlgs\Documents\Project_fraud_2\sestuple_datasets\dev_set_classic_sestuple"
    training_set = TupleDataset.load(TRAINING_SET_PATH)
    dev_set = TupleDataset.load(DEV_SET_PATH)
    
    # normalizzo
    normalizer = StandardNormalizer()
    normalizer.fit(training_set.source_dataset.records)
    training_set.source_dataset.apply_tranformer(normalizer)
    dev_set.source_dataset.apply_tranformer(normalizer)

    # estraggo una sola feature dai due dataset
    def f(record): return [record.seq_dd[1]]
    one_feature_training_set = [training_set.get_custom_sample(
        f, i) for i in range(len(training_set))]
    one_feature_dev_set = [dev_set.get_custom_sample(
        f, i) for i in range(len(dev_set))]
