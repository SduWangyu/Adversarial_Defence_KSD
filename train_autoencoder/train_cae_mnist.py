from torch import nn
import torch
from utils import misc as ksd
import matplotlib.pyplot as plt


class ConvAutoEncoder_MNIST(nn.Module):
    def __init__(self):
        super(ConvAutoEncoder_MNIST, self).__init__()
        self.autocode = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(16, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1),
            nn.ConvTranspose2d(8, 16, kernel_size=3, stride=2), nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, kernel_size=5, stride=3, padding=1), nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, kernel_size=2, stride=2, padding=1), nn.ReLU(),
        )

    def forward(self, x):
        output = self.autocode(x)
        return output


device = 'cuda' if torch.cuda.is_availabel() else 'cpu'


def show_images(decode_images, x_test):
    """
    plot the images.
    :param decode_images: the images after decoding
    :param x_test: testing data
    :return:
    """
    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        ax.imshow(x_test[i].reshape(28, 28, 1))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax = plt.subplot(2, n, i + 1 + n)
        ax.imshow(decode_images[i].detach().numpy().reshape(28, 28, 1))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


if __name__ == '__main__':
    X = torch.rand((1, 1, 28, 28))
    net = ConvAutoEncoder_MNIST().autocode
    for layer in net:
        print(layer.__class__.__name__, 'output shape:\t', X.shape)
        X = layer(X)

    batch_size = 64
    train_iter, test_iter = ksd.load_data_mnist(batch_size)
    lr, num_epochs = 0.001, 20

    ksd.train_autoencoder(net, train_iter, test_iter, num_epochs, lr, ksd.try_gpu())
    torch.save(net.state_dict(), '../models/autoencoders/conv_autoencoder_mnist.pth')
