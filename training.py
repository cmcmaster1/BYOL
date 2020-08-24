#%%
import os
import argparse
import multiprocessing
from pathlib import Path
from PIL import Image

import torch
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, Dataset
from torch import nn

from byol import BYOL
from data import ImagesDataset, BYOLImagesDataset
import pytorch_lightning as pl

data_path = Path('../../Data/hep2/uq')
all_images = Path.joinpath(data_path, 'all_images')
labelled = Path.joinpath(data_path, 'laeblled')
labelled_test = Path.joinpath(data_path, 'labelled_test')


#%%
# constants
BATCH_SIZE = 256
EPOCHS     = 1
LR         = 3e-4
NUM_GPUS   = 1
IMAGE_SIZE = 90
IMAGE_EXTS = ['.jpg', '.png', '.jpeg']
NUM_WORKERS = multiprocessing.cpu_count()

# base model
base_model = models.resnet18()

# pytorch lightning data module
class BYOLDataModule(pl.LightningDataModule):
    def __init__(self, folder, image_size, batch_size, exts):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.batch_size = batch_size
        self.exts = exts

        self.dims = (1, self.image_size, self.image_size)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            train_data = BYOLImagesDataset(
                folder=self.folder, 
                image_size=self.image_size,
                exts=self.exts)
            self.train_data = train_data


    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size)

# pytorch lightning module
class SelfSupervisedLearner(pl.LightningModule):
    def __init__(self, net, **kwargs):
        super().__init__()
        self.learner = BYOL(net, **kwargs)

    def forward(self, images):
        return self.learner(images)

    def training_step(self, images, _):
        loss = self.forward(images)
        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=LR)

    def on_before_zero_grad(self, _):
        self.learner.update_moving_average()

dm = BYOLDataModule(
    folder=all_images, 
    image_size=90, 
    batch_size=BATCH_SIZE,
    exts=IMAGE_EXTS)

model = SelfSupervisedLearner(
        base_model,
        image_size = IMAGE_SIZE,
        hidden_layer = 'layer3',
        projection_size = 256,
        projection_hidden_size = 1024,
        moving_average_decay = 0.99
        )
#%%
trainer = pl.Trainer(gpus=-1, max_epochs=EPOCHS)
trainer.fit(model, datamodule=dm)
# %%
