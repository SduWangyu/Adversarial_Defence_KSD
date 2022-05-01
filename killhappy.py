import torch
from core.autoencoders import ResAutoEncoderCIFAR100 as rae

net = rae()
X = torch.rand(size=(1, 3, 32, 32))
net(X)
