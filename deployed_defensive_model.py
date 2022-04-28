from torch import nn
import autoencoders as aes
import classifiers as clfs
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch
import utils as ksd

class Defence_MNIST(nn.Module):
    def __init__(self):
        super(Defence_MNIST, self).__init__()
        self.autoencoder = aes.ConvAutoEncoder(encoded_space_dim=10, fc2_input_dim=128)
        self.autoencoder.load_state_dict(torch.load('./models/autoencoders/conv_autoencoder_mnist.pth'))
        self.autoencoder.eval()
        self.classifier = clfs.Classifer_MNIST()
        self.classifier.load_state_dict(torch.load('./models/classifiers/classifier_mnist.pth'))
        self.classifier.eval()

    def forward(self, x):
        ths = 0.01
        reg_x = self.autoencoder(x)
        ori_pre = self.classifier(x)
        rec_pre = self.classifier(reg_x.reshape(x.shape))
        if ksd.jenson_shannon_divergence(ori_pre, rec_pre).sum() > ths:
            return ori_pre, True
        else:
            return ori_pre, False


if __name__ == '__main__':
    net = Defence_MNIST()
    data = datasets.MNIST(
        root='./data',
        train=False,
        transform=transforms.ToTensor()
    )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for X, y in data:
        X = torch.unsqueeze(X, 0)
        net.to(device)
        X = X.to(device)
