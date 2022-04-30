import torch
import torchvision.models as models
from torchvision import datasets
from torchvision.transforms import ToTensor, Resize, Compose
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_availabel() else 'cpu'
    alexnet = models.alexnet(pretrained=True).eval()
    mnist_train_dataset = datasets.MNIST(
        root='../data',
        train=True,
        download=True,
        transform=ToTensor()
    )
    for x in mnist_train_dataset:
        output = alexnet(x.to(device))
        print(output)

