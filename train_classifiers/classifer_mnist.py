import torch
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose
from torch import nn


class Classifer_MNIST(nn.Module):
    def __init__(self, num_class: int = 10) -> None:
        super(Classifer_MNIST, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Linear(1024, 200),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(200, 10),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        output = model(X.to(device)).to(device)
        loss = loss_fn(output, y.to(device))
        # Backpropagation

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

#
# def test_loop(dataloader, model, loss_fn):
#     num_batches = len(dataloader)
#     test_loss, correct = 0, 0
#     with torch.no_grad():
#         for X, y in dataloader:
#             output = model(X.to(device)).to(device)
#             test_loss += loss_fn(output, y.to(device)).item()
#
#     test_loss /= num_batches
#     print(f"Test Error: Avg loss: {test_loss:>8f} \n")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X.to(device)).cpu()
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if __name__ == '__main__':

    test_mnist_dataset = datasets.MNIST(
        root='../data',
        train=False,
        download=True,
        transform=ToTensor(),
    )


    model = Classifer_MNIST()
    model.load_state_dict(torch.load('../models/classifiers/classifier_minst.pth'))
    model.eval()
    model = model.to(device)

    test_dataloader = DataLoader(test_mnist_dataset, batch_size=128, shuffle=True)
    loss_fn = nn.CrossEntropyLoss()
    test_loop(test_dataloader, model, loss_fn)

    # train_mnist_dataset = datasets.MNIST(
    #     root='../data',
    #     train=True,
    #     download=True,
    #     transform=ToTensor(),
    # )
    #
    # test_mnist_dataset = datasets.MNIST(
    #     root='../data',
    #     train=False,
    #     download=True,
    #     transform=ToTensor(),
    # )
    #
    # epochs = 50
    # model = Classifer_MNIST().to(device)
    # learning_rate = 0.01
    # batch_size = 128
    # # Initialize the loss function
    # loss_fn = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    # train_dataloader = DataLoader(train_mnist_dataset, batch_size=batch_size, shuffle=True)
    # test_dataloader = DataLoader(test_mnist_dataset, batch_size=batch_size, shuffle=True)
    # for epoch in range(epochs):
    #     print(f"Epoch {epoch + 1}\n-------------------------------")
    #     train_loop(train_dataloader, model, loss_fn, optimizer)
    #     test_loop(test_dataloader, model, loss_fn)
    #
    # torch.save(model.state_dict(), '../models/classifiers/classifier_minst.pth')

