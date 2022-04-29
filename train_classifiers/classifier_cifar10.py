import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, RandomHorizontalFlip
from torch import nn


class Classifer_CIFAR10(nn.Module):
    def __init__(self, num_class: int = 10) -> None:
        super(Classifer_CIFAR10, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=3, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=3, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(96, 192, kernel_size=3, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 192, kernel_size=3, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 10, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Linear(1210, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 200),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(200, 10),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
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
            pred = model(X.to(device))
            test_loss += loss_fn(pred, y.to(device)).item()
            correct += (pred.argmax(1) == y.to(device)).cpu().type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


device = 'cuda' if torch.cuda.is_availabel() else 'cpu'
if __name__ == '__main__':
    totensor_transform = Compose([RandomHorizontalFlip(), ToTensor()])
    train_mnist_dataset = datasets.CIFAR10(
        root='../data',
        train=True,
        download=True,
        transform=totensor_transform,
    )
    for x in train_mnist_dataset:
        print(x[0].shape)
        break
    test_mnist_dataset = datasets.CIFAR10(
        root='../data',
        train=False,
        download=True,
        transform=totensor_transform,
    )
    #
    epochs = 350
    model = Classifer_CIFAR10().to(device)
    learning_rate = 0.01
    batch_size = 32
    # Initialize the loss function
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    train_dataloader = DataLoader(train_mnist_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_mnist_dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)

    torch.save(model.state_dict(), '../models/classifiers/classifier_cifar10.pth')
    # X = torch.rand(size=(1, 3, 32, 32))
    # for layer in Classifer_CIFAR10().features:
    #     X = layer(X)
    #     print(layer.__class__.__name__, 'output shape:\t', X.shape)