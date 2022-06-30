from nbformat import write
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import dataset, generator, discriminator, config
from utils import *
import torchvision.utils as vutils
from torchsummary import summary
from tqdm import tqdm
import os
import random

#set manual seed to a constant get a consistent output
manualSeed = random.randint(1, 2120)
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: ", device)
cfg = config.config
if not os.path.exists('ckpt'):
    os.makedirs('ckpt')
if not os.path.exists('output'):
    os.makedirs('output')

G = generator.Generator(cfg.num_classes, cfg.noise_size, cfg.num_g_filters).to(device)
G.apply(weights_init)
D = discriminator.Discriminator(cfg.num_classes, cfg.num_d_filters).to(device)
D.apply(weights_init)
# summary(G, (cfg.noise_size, cfg.num_classes))
# summary(D, (cfg.num_classes,))

dataset = dataset.get_dataset()
dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, drop_last=True)

criterion = nn.BCELoss()
optimizer_G = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
# schedule_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=50, gamma=0.1)
optimizer_D = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
# schedule_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=50, gamma=0.1)

real_label = 1.
fake_label = 0.
fixed_noise = torch.randn(16, cfg.noise_size, 1, 1, device=device)
best_loss = 100.

for epoch in range(cfg.num_epochs + 1):
    G.train()
    D.train()
    for real_imgs, _ in tqdm(dataloader):
        real_imgs = real_imgs.to(device)
        noise = torch.randn(cfg.batch_size, cfg.noise_size, 1, 1, device=device)
        fake_imgs = G(noise)
        dis_fake_out = D(fake_imgs.detach())

        # Train D
        D.zero_grad()
        dis_real_out = D(real_imgs)
        if cfg.loss == 'BCE':

            labels = torch.full((cfg.batch_size,), real_label, device=device)
            dis_real_loss = criterion(dis_real_out, labels)
            dis_real_loss = dis_real_loss.mean()
            dis_real_loss.backward()

            labels.fill_(fake_label)
            dis_fake_loss = criterion(dis_fake_out, labels)
            dis_fake_loss = dis_fake_loss.mean()
            dis_fake_loss.backward()
            dis_loss = dis_real_loss + dis_fake_loss

        elif cfg.loss == 'WGAN':
            dis_real_loss = -torch.mean(dis_real_out)
            dis_fake_loss = torch.mean(dis_fake_out)
            dis_loss = dis_real_loss + dis_fake_loss
            dis_loss.backward()
            
        optimizer_D.step()

        # Train G
        G.zero_grad()
        labels.fill_(real_label)
        dis_fake_out = D(fake_imgs)
        if cfg.loss == 'BCE':
            gen_loss = criterion(dis_fake_out, labels)
            gen_loss = gen_loss.mean()
        elif cfg.loss == 'WGAN':
            gen_loss = -torch.mean(dis_fake_out)

        gen_loss.backward()
        optimizer_G.step()

    # write to training log
    with open('train_log.txt', 'a') as f:
        f.write('Epoch: {} \t Generator Loss: {} \t Discriminator Loss: {}\n'.format(epoch, gen_loss, dis_loss))
    print("Epoch: {}/{} G_loss: {} D_loss: {}".format(epoch, cfg.num_epochs, gen_loss, dis_loss))
    # schedule_G.step()
    # schedule_D.step()
    if best_loss > dis_loss * cfg.alpha + gen_loss:
        best_loss = dis_loss * cfg.alpha + gen_loss
        torch.save(G.state_dict(), "ckpt/G.pth")
        torch.save(D.state_dict(), "ckpt/D.pth")
        
    if epoch % 50 == 0:
        G.eval()
        fake_imgs = G(fixed_noise)
        vutils.save_image(fake_imgs.detach(), "output/{}.png".format(epoch), normalize=True)






