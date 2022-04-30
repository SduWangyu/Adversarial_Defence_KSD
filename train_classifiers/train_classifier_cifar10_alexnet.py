import torch
from torch import nn
from utils import misc as ksd

Classifier_CIFAR100 = nn.Sequential(
            # 这里，我们使用一个11*11的更大窗口来捕捉对象。
            # 同时，步幅为4，以减少输出的高度和宽度。
            # 另外，输出通道的数目远大于LeNet
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
            nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 使用三个连续的卷积层和较小的卷积窗口。
            # 除了最后的卷积层，输出通道的数量进一步增加。
            # 在前两个卷积层之后，汇聚层不用于减少输入的高度和宽度
            nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Flatten(),
            # 这里，全连接层的输出数量是LeNet中的好几倍。使用dropout层来减轻过拟合
            nn.Linear(6400, 4096), nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096), nn.ReLU(),
            nn.Dropout(p=0.5),
            # 最后是输出层。由于这里使用Fashion-MNIST，所以用类别数为10，而非论文中的1000
            nn.Linear(4096, 10),
        )


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        optimizer.zero_grad()
        # Compute prediction and loss
        X, y = X.to(device), y.to(device)
        output = model(X)
        loss = loss_fn(output, y.to(device))
        # Backpropagation
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

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)



device = 'cuda' if torch.cuda.is_availabel() else 'cpu'
if __name__ == '__main__':
    batch_size = 32
    train_iter, test_iter = ksd.load_data_cifar_10(batch_size, resize=224)
    lr, num_epochs = 0.01, 100
    net = Classifier_CIFAR100
    ksd.train_classifier(net, train_iter, test_iter, num_epochs, lr, ksd.try_gpu())
    torch.save(net.state_dict(), '../models/classifiers/classifer_cifar10.pth')