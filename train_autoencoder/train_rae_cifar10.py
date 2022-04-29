import torch
from utils import utils as ksd
from core.autoencoders import ResAutoEncoderCIFAR10


if __name__ == '__main__':

    loss_fn = torch.nn.MSELoss()
    lr = 1e-3
    torch.manual_seed(0)
    d = 4
    epochs = 50
    train_loader, test_loader = ksd.load_data_cifar_10(batch_size=64)
    net = ResAutoEncoderCIFAR10()
    ksd.train_autoencoder_and_save(net, train_loader, test_loader, epochs, lr, noise_factor=0.1, device=ksd.try_gpu(), model_name='../models/autoencoders/res_autoencoder_cifar10.pth')
    for X, y in test_loader:
        ksd.show_images(net(X.to("cuda")).to('cpu'), X)
        break

