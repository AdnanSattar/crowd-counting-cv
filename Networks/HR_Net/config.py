import os
import time

import torch
from easydict import EasyDict as edict

# init
__C = edict()
cfg = __C

# ------------------------------TRAIN------------------------
__C.SEED = 1  # random seed,  for reproduction


__C.NET = "HR_Net"
__C.PRE_HR_WEIGHTS = "./Networks/HR_Net/hrnetv2_w48_imagenet_pretrained.pth"
