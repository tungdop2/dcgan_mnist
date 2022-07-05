import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from dataset import get_dataset
from discriminator import Discriminator
from generator import Generator
from utils import weights_init
from config import config

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=64, num_workers=4):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    def setup(self, stage=None):
        self.dataset = get_dataset()
        
    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, drop_last=True)
    
    def __len__(self):
        return len(self.dataset)

class MNISTGAN(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.G = Generator(cfg.num_classes, cfg.noise_size, cfg.num_g_filters)
        self.D = Discriminator(cfg.num_classes, cfg.num_d_filters)
        self.G.apply(weights_init)
        self.D.apply(weights_init)
        self.fixed_noise = torch.randn(16, cfg.noise_size, 1, 1).to(device)

    def forward(self, x):
        return self.G(x)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        real_imgs, _ = batch
        noise = torch.randn(real_imgs.size(0), self.cfg.noise_size, 1, 1).to(device)

        if optimizer_idx == 0:
            fake_imgs = self(noise)
            fake_labels = torch.ones(real_imgs.size(0)).to(device)
            g_loss = self.adversarial_loss(self.D(fake_imgs), fake_labels)

            return {'loss': g_loss, 'log': {'g_loss': g_loss}, 'progress_bar': {'g_loss': g_loss}}
        else:
            fake_labels = torch.zeros(real_imgs.size(0)).to(device)
            fake_imgs = self(noise)
            d_loss_fake = self.adversarial_loss(self.D(fake_imgs.detach()), fake_labels)
            real_labels = torch.ones(real_imgs.size(0)).to(device)
            d_loss_real = self.adversarial_loss(self.D(real_imgs), real_labels)
            d_loss = d_loss_fake * self.cfg.alpha + d_loss_real * (1 - self.cfg.alpha)

            return {'loss': d_loss, 'log': {'d_loss': d_loss}, 'progress_bar': {'d_loss': d_loss}}

    def configure_optimizers(self):
        optimizer_G = torch.optim.Adam(self.G.parameters(), lr=0.0002, betas=(0.5, 0.999))
        optimizer_D = torch.optim.Adam(self.D.parameters(), lr=0.0002, betas=(0.5, 0.999))

        scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=50, gamma=0.95)
        scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=50, gamma=0.95)

        return [optimizer_G, optimizer_D], [scheduler_G, scheduler_D]

    def on_epoch_end(self):
        fake_imgs = self.G(self.fixed_noise)
        grids = torchvision.utils.make_grid(fake_imgs, nrow=4, padding=2, normalize=True)
        if self.current_epoch % 50 == 0:
            torchvision.utils.save_image(grids, os.path.join('./output', 'fake_imgs_epoch_{}.png'.format(self.current_epoch)))
        self.logger.experiment.add_image('fake_images', grids, self.current_epoch)

        torch.save(self.G.state_dict(), os.path.join('./ckpt', 'G.pth'))
        torch.save(self.D.state_dict(), os.path.join('./ckpt', 'D.pth'))

cfg = config
dm = MNISTDataModule(batch_size=cfg.batch_size, num_workers=cfg.num_workers)
model = MNISTGAN(cfg)

os.makedirs('./logger', exist_ok=True)
os.makedirs('./output', exist_ok=True)
os.makedirs('./ckpt', exist_ok=True)
logger = TensorBoardLogger(save_dir='./logger', name='mnist_gan')
trainer = pl.Trainer(gpus=1, max_epochs=cfg.num_epochs, logger=logger, checkpoint_callback=False)
trainer.fit(model, dm)


