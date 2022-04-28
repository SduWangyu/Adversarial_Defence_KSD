import torch
import utils as ksd
from classifiers import Classifer_MNIST
from autoencoders import ConvAutoEncoder
if __name__ == '__main__':
    batch_size = 64
    train_iter, test_iter = ksd.load_data_mnist(batch_size)
    lr, num_epochs = 0.01, 50
    net = Classifer_MNIST()
    ae = ConvAutoEncoder(encoded_space_dim=10, fc2_input_dim=128)
    ae.load_state_dict(torch.load('../models/autoencoders/conv_autoencoder_mnist.pth'))
    ae.eval()
    ksd.train_classifier_by_autoencoder_and_save(net, ae, train_iter, test_iter, num_epochs, lr, ksd.try_gpu())
    torch.save(net.state_dict(), '../models/classifiers/classifier_mnist.pth')
