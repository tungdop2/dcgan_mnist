from easydict import EasyDict as edict

config = edict()
config.num_epoch = 200
config.num_classes = 1
config.num_channels = 1
config.batch_size = 2048
config.noise_size = 100
config.image_size = 28
config.num_workers = 4