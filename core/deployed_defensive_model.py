from torch import nn
from core import classifiers as clfs, autoencoders as aes
from torchvision import datasets, transforms
import torch



class DefenceMNIST(nn.Module):
    def __init__(self):
        super(DefenceMNIST, self).__init__()
        self.autoencoder = aes.ConvAutoEncoderMNIST(encoded_space_dim=10, fc2_input_dim=128)
        self.autoencoder.load_exist()
        self.autoencoder.eval()
        self.classifier = clfs.ClassiferMNIST()
        self.classifier.load_exist()
        self.classifier.eval()

    def forward(self, x):
        ths = 61.1826
        reg_x = self.autoencoder(x)
        # ori_pre = self.classifier(x)
        rec_pre = self.classifier(reg_x)
        if torch.dist(x, reg_x, p=1) > ths:
            return rec_pre, True
        else:
            return rec_pre, False


class DefenceCIFAR10(nn.Module):
    def __init__(self):
        super(DefenceCIFAR10, self).__init__()
        self.autoencoder = aes.ConvAutoEncoderCIFAR10()
        self.autoencoder.load_exist(path=r'C:\Users\Fra\Desktop\Adversarial_Defence_KSD\data\models_trained\autoencoders\cov_cifar10_colab.pth')
        self.autoencoder.eval()
        self.classifier = clfs.ClassifierCIFAR10()
        self.classifier.load_exist()
        self.classifier.eval()

    def forward(self, x):
        ths = 461.302
        reg_x = self.autoencoder(x)
        # ori_pre = self.classifier(x)
        rec_pre = self.classifier(reg_x)
        if torch.dist(x, reg_x, p=1) > ths:
            return rec_pre, True
        else:
            return rec_pre, False


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