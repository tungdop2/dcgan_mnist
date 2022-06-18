import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import dataset, generator, discriminator, config
from utils import weights_init
import torchvision.utils as vutils
from tqdm import tqdm
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cfg = config.config

G = generator.Generator(cfg.num_classes, cfg.noise_size, cfg.num_channels).to(device)
G.apply(weights_init)
D = discriminator.Discriminator(cfg.num_classes, cfg.num_channels).to(device)
D.apply(weights_init)

dataset = dataset.get_dataset()
dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, drop_last=True)

criterion = nn.BCELoss()
optimizer_G = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
# schedule_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=50, gamma=0.1)
optimizer_D = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
# schedule_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=50, gamma=0.1)

real_label = 1.
fake_label = 0.

for epoch in range(cfg.num_epoch):
    for real_imgs, _ in tqdm(dataloader):
        real_imgs = real_imgs.to(device)
        labels = torch.full((cfg.batch_size,), real_label, device=device)

        # Train D
        D.zero_grad()
        dis_real_out = D(real_imgs)
        dis_real_loss = criterion(dis_real_out, labels)
        dis_real_loss.backward()
        dis_real_loss = dis_real_loss.mean()

        noise = torch.randn(cfg.batch_size, cfg.noise_size, 1, 1, device=device)
        fake_imgs = G(noise)
        labels.fill_(fake_label)
        dis_fake_out = D(fake_imgs.detach())
        dis_fake_loss = criterion(dis_fake_out, labels)
        dis_fake_loss.backward()
        dis_fake_loss = dis_fake_loss.mean()

        dis_loss = dis_real_loss + dis_fake_loss
        optimizer_D.step()

        # Train G
        G.zero_grad()
        dis_fake_out = D(fake_imgs)
        dis_fake_loss = criterion(dis_fake_out, labels)
        gen_loss = criterion(dis_fake_out, labels)
        gen_loss.backward()
        gen_loss = gen_loss.mean()
        optimizer_G.step()

    print("Epoch: {}/{} \nG_loss: {} with lr: {} \nD_loss: {} with lr: {}".format(epoch, cfg.num_epoch, gen_loss, optimizer_G.param_groups[0]['lr'], dis_loss, optimizer_D.param_groups[0]['lr']))
    # schedule_G.step()
    # schedule_D.step()
    if epoch % 20 == 0:
        if not os.path.exists('ckpt'):
            os.makedirs('ckpt')
        if not os.path.exists('output'):
            os.makedirs('output')
        torch.save(G.state_dict(), "ckpt/G.pth")
        torch.save(D.state_dict(), "ckpt/D.pth")
        vutils.save_image(fake_imgs.data[:16], "output/epoch_{}.png".format(epoch), normalize=True)
        # vutils.save_image(real_imgs.data[:16], "output/real_epoch_{}.png".format(epoch), normalize=True)






