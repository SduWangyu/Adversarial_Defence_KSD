import torch
from torch.utils.data import DataLoader
import core.autoencoders as ae
import utils.constants
from core.custom_dataset import CustomDataset
import core.classifiers
from core.custom_dataset import load_data
from train import train_autoencoder
from torchvision import transforms
from core.defence import decrease_color_depth, denoising_by_Gaussian
import matplotlib.pyplot as plt
from utils import constants
from utils.misc import try_gpu
from utils.image_operator import show_one_images

if __name__ == "__main__":
    # net = ae.ConvLargeAutoEncoderCIFAR10()
    # trans_train = transforms.Compose([
    #     # transforms.RandomCrop(32, padding=4),
    #     # transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize(constants.cifar10_test_mean, constants.cifar10_test_std)])
    # trans_test = transforms.Compose([
    #     transforms.ToTensor()
    #
    # ])
    # train_iter, test_iter = load_data("cifar10", batch_size=128, transforms_train=trans_train, transforms_test=trans_test)
    # # # train_classifier(net, train_iter, test_iter, num_epochs=2, lr=0.01,
    # # #                  device=try_gpu(), model_name='./data/models_trained/classifiers/resnet18_cifar10.pth')
    # train_autoencoder(net, train_iter, test_iter, num_epochs=400, lr=0.1,
    #                   device=try_gpu(), model_name='./data/models_trained/autoencoders/cov_cifar10_test_32.pth')
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
    device = try_gpu()
    # net = ae.ConvLargeAutoEncoderCIFAR10()
    # net.to(device)
    # net.load_exist(path='data/models_trained/autoencoders/cov_cifar10_test_64.pth')
    net = denoising_by_Gaussian
    # dataset_adv = torch.load(f"./data/validation_data/validation_data_mnist_cw_10_adv.pt")
    # data_adv_iter = DataLoader(dataset_adv, batch_size=10)
    dataset_org = torch.load(f"./data/validation_data/validation_data_cifar10_cw_0_org_no_normalize.pt")
    data_org_iter = DataLoader(dataset_org, batch_size=1)

    # for (X, _), (X_adv, _) in zip(data_org_iter, data_adv_iter):
    #     X, X_adv = X.to(device), X_adv.to(device)
    #     # X_rec = net(X).cpu()
    #     X_adv_rec = net(X).cpu()
    #     show_image(X.cpu(), X_adv.cpu(), X_adv_rec)
    #     break
    p = 2
    pas = 0
    for X, _ in data_org_iter:
        pas += 1
        if pas <= 2:
            continue
        # ax = plt.subplot(2, n,  i + 1)
        # # print(tmp_rec.shape, tmp_org.shape, tmp_adv.shape)
        # ax.imshow(X.squeeze(0).detach().clone().numpy().transpose(1, 2, 0))
        #
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)
        # constants.shift = i
        ax = plt.subplot(2, 8, pas - 2)
        # print(tmp_rec.shape, tmp_org.shape, tmp_adv.shape)
        ax.imshow(X.squeeze(0).detach().clone().numpy().transpose(1, 2, 0))

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, 8, pas + 8 - 2)
        tmp_img = net(X)
        # print(tmp_img.shape)
        tmp_img = tmp_img.squeeze(0).detach().clone().numpy().transpose(1, 2, 0)
        # print(tmp_rec.shape, tmp_org.shape, tmp_adv.shape)
        ax.imshow(tmp_img)

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if pas == 10:
            break
    plt.show()
