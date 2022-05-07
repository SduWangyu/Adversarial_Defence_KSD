import torch
from torch.utils.data import DataLoader
import core.autoencoders as ae
from utils import constants
from core.custom_dataset import CustomDataset
import core.classifiers
from core.custom_dataset import load_data
from train import train_autoencoder
from torchvision import transforms
from utils.misc import try_gpu
from utils.image_operator import show_images_ae as show_image

if __name__ == "__main__":
    net = ae.ConvLargeAutoEncoderCIFAR10()
    trans_train = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(constants.cifar10_test_mean, constants.cifar10_test_std)])
    trans_test = transforms.Compose([
        transforms.ToTensor()

    ])
    train_iter, test_iter = load_data("cifar10", batch_size=128, transforms_train=trans_train, transforms_test=trans_test)
    # # train_classifier(net, train_iter, test_iter, num_epochs=2, lr=0.01,
    # #                  device=try_gpu(), model_name='./data/models_trained/classifiers/resnet18_cifar10.pth')
    train_autoencoder(net, train_iter, test_iter, num_epochs=400, lr=0.01,
                      device=try_gpu(), model_name='./data/models_trained/autoencoders/cov_cifar10_test_32.pth')
    # # net = core.classifiers.ResNet18CIFAR10()
    # net.load_exist(path="data/models_trained/classifiers/resnet18_cifar10.pth")
    # correct = 0
    # total_data = 0
    # for X, y in test_iter:
    #     res_y = net(X).argmax(1)
    #     for i in range(len(y)):
    #         correct += (y[i] == res_y[i])
    #         total_data += 1
    #         if total_data == 5000:
    #             break
    #         if not total_data % 100:
    #             print(total_data)
    #     else:
    #         continue
    #     break
    # print(total_data, correct, correct / total_data)
    # device = try_gpu()
    # net = ae.ConvAutoEncoderCIFAR10()
    # net.to(device)
    # net.load_exist(path='data/models_trained/autoencoders/cov_cifar10_test.pth')
    # dataset_adv = torch.load(f"./data/validation_data/validation_data_cifar10_cw_10_adv.pt")
    # data_adv_iter = DataLoader(dataset_adv, batch_size=10)
    # dataset_org = torch.load(f"./data/validation_data/validation_data_cifar10_cw_10_org.pt")
    # data_org_iter = DataLoader(dataset_org, batch_size=10)
    #
    # for (X, _), (X_adv, _) in zip(data_org_iter, data_adv_iter):
    #     X, X_adv = X.to(device), X_adv.to(device)
    #     # X_rec = net(X).cpu()
    #     X_adv_rec = net(X).cpu()
    #     show_image(X.cpu(), X_adv.cpu(), X_adv_rec)
    #     break

