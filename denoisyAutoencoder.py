import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np




def add_noise(train_data, test_data):
    train_data_noisy = train_data + NOISE_FACTOR * np.random.normal(loc=0.0, scale=1.0, size=train_data.shape)
    test_data_noisy = test_data + NOISE_FACTOR * np.random.normal(loc=0.0, scale=1.0, size=test_data.shape)
    train_data_noisy = np.clip(train_data_noisy, 0., 1.).to(torch.float32)
    test_data_noisy = np.clip(test_data_noisy, 0., 1.).to(torch.float32)
    return train_data_noisy, test_data_noisy


class Denoise_AutoEncoder_MNIST(nn.Module):
    def __init__(self):
        super(Denoise_AutoEncoder_MNIST, self).__init__()
        self.autocode = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
            nn.ReLU(),
            nn.Linear(10, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.ReLU(),
        )

    def forward(self, x):
        output = self.autocode(x)
        return output

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
        ax = plt.subplot(2, n, i+1)
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


def train_loop(dataloader, noise_dataloader ,model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    batch = 0
    for X, X_noisy in zip(dataloader, noise_dataloader):
        X = X.to(device)
        X_noisy = X_noisy.to(device)
        # Compute prediction and loss
        output = model(X_noisy)
        loss = loss_fn(output, X)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch+=1
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")



def test_loop(dataloader, noise_dataloader,model, loss_fn):
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, X_noise in zip(dataloader, noise_dataloader):
            X = X.to(device)
            X_noise = X_noise.to(device)
            output = model(X_noise).to(device)
            test_loss += loss_fn(output, X).item()

    test_loss /= num_batches
    print(f"Test Error: Avg loss: {test_loss:>8f} \n")





if __name__ == '__main__':
    NOISE_FACTOR = 0.1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Denoise_AutoEncoder_MNIST()
    model.to(device)

    print(device)
    learning_rate = 0.001
    batch_size = 32
    # Initialize the loss function
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    ).data

    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    ).data

    # train_noisy_data, test_noisy_data = add_noisy(train_data, test_data)
    x_train = torch.tensor(train_data.reshape((train_data.shape[0], -1)), dtype=torch.float32)/255.

    x_test = torch.tensor(test_data.reshape((test_data.shape[0], -1)), dtype= torch.float32)/255.
    x_train_noisy, x_test_noisy = add_noise(x_train, x_test)
    train_dataloader = DataLoader(x_train, batch_size=64)
    train_noisy_dataloader = DataLoader(x_train_noisy, batch_size=64)
    test_dataloader = DataLoader(x_test, batch_size=64)
    test_noisy_dataloader = DataLoader(x_test_noisy, batch_size=64)
    show_images(x_test_noisy, x_test)
    epochs = 20
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_dataloader, train_noisy_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, test_noisy_dataloader, model, loss_fn)
    torch.save(model.state_dict(), './models/autoencoders/denoise_autoencoder_mnist.pth')
    show_images(model(x_test_noisy.to(device)).to('cpu'), x_test)
