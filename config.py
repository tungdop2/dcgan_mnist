from easydict import EasyDict as edict

config = edict()
config.num_epochs = 300
config.num_classes = 1
config.num_g_filters = 64
config.num_d_filters = 64
config.batch_size = 2048
config.noise_size = 128
config.image_size = 28
config.num_workers = 2
config.alpha = 0.2
config.loss = 'BCE'