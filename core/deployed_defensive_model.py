from torch import nn

import utils.evaluate
from core import classifiers as clfs, autoencoders as aes
from torchvision import datasets, transforms
import torch
import utils.evaluate as evt


class DefenceMNIST(nn.Module):
    def __init__(self, ths, denoising=[]):
        super(DefenceMNIST, self).__init__()
        self.autoencoder = aes.ConvAutoEncoderMNIST(encoded_space_dim=10, fc2_input_dim=128)
        self.autoencoder.load_exist()
        self.autoencoder.eval()
        self.denoising = denoising
        self.classifier = clfs.ClassiferMNIST()
        self.classifier.load_exist()
        self.classifier.eval()
        self.ths = ths

    def forward(self, x):
        den_x = x
        # print(x.shape)
        org_pre = self.classifier(x)
        for denoise in self.denoising:
            den_x = denoise(den_x)
        reg_x = den_x
        # reg_x = self.autoencoder(den_x)
        # ori_pre = self.classifier(x)
        rec_pre = self.classifier(reg_x)
        # print(org_pre.shape, rec_pre.shape)
        # print(utils.evaluate.jenson_shannon_divergence(org_pre, rec_pre))
        if utils.evaluate.jenson_shannon_divergence(org_pre, rec_pre) > self.ths:
            return rec_pre, True
        else:
            return rec_pre, False


class DefenceCIFAR10(nn.Module):
    def __init__(self, thr, denoising=[]):
        super(DefenceCIFAR10, self).__init__()
        # self.autoencoder = aes.ConvLargeAutoEncoderCIFAR10()
        # self.autoencoder.load_exist(path=r'../data\models_trained\autoencoders\cov_cifar10_test_64_liner.pth')
        # self.autoencoder.eval()
        self.denoising = denoising
        self.classifier = clfs.ResNet18CIFAR10()
        self.classifier.load_exist()
        self.classifier.eval()
        self.thr = thr

    def forward(self, x):
        # ths = 461.302
        # return self.classifier(x), False
        den_x = x
        for denoise in self.denoising:
            # print(den_x.data.device)
            den_x = denoise(den_x)
        reg_x = den_x
        # reg_x = self.autoencoder(den_x)
        # ori_pre = self.classifier(x)
        rec_pre = self.classifier(reg_x)
        org_pre = self.classifier(x)
        if utils.evaluate.jenson_shannon_divergence(org_pre, rec_pre) > self.thr:
            return rec_pre, True
        else:
            return rec_pre, False
        # if torch.dist(x, reg_x, p=1) > ths:
        #     return rec_pre, True
        # else:
        #     return rec_pre, False


if __name__ == '__main__':
    net = DefenceMNIST()
    data = datasets.MNIST(
        root='./data',
        train=False,
        transform=transforms.ToTensor()
    )
    from core.attack_method import fgsm
    device = 'cuda' if torch.cuda.is_availabel() else 'cpu'
    for X, y in data:
        net.to(device)
        X = X.to(device)
        # print(y)
        X_adv = fgsm(net.classifier, X, e=0.01, device=device, max_iterations=100, target_label=(y+1)%10)
        if X_adv is None:
            continue
        # print(X_adv)

        net(torch.unsqueeze(X, 0))
        net(torch.unsqueeze(X_adv, 0))
        break