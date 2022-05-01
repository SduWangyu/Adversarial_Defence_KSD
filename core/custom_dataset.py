import torch
import torchvision
from torch.utils import data
import sys   #使用sys库添加路径
sys.path.append("..")   #这句是为了导入_config

import attack_method as am
from torch.utils.data import random_split
import classifiers as clfs
from utils import constants
from utils.misc import try_gpu
from torchvision import transforms

dataloader_workers = 4
data_root = "../data/"


class CustomDataset(data.Dataset):
    def __init__(self, data_size):
        super(CustomDataset, self).__init__()
        self.x = torch.zeros(data_size)
        self.y = torch.zeros(data_size[0])

    def __getitem__(self, index):
        img, label = self.x[index], self.y[index]
        return img, label

    def __len__(self):
        return self.x.shape[0]

    def update_data(self, index, data_in):
        self.x[index] = data_in[0]
        self.y[index] = data_in[1]


def load_data(name, batch_size, val_type=0, val_account=0.1, transforms_train=None, transforms_test=None):
    """
    :param transforms_test:
    :param transforms_train:
    :param val_account:
    :param val_type:
    :param name: dataset name
    :param batch_size:
    :return:
    """

    if name == "mnist":
        method = torchvision.datasets.MNIST
    elif name == "cifar10":
        method = torchvision.datasets.CIFAR10
    elif name == "cifar100":
        method = torchvision.datasets.CIFAR100
    elif name == "imagenet":
        method = torchvision.datasets.ImageNet
    else:
        raise Exception("Data name error!")

    train_data = method(
        root=data_root,
        train=True,
        download=True,
        transform=transforms_train
    )

    if val_type == 0:
        test_data = method(root=data_root,
                           train=False,
                           download=True,
                           transform=transforms_test)

        return (data.DataLoader(train_data, batch_size, shuffle=True,
                                num_workers=dataloader_workers),
                data.DataLoader(test_data, batch_size, shuffle=False,
                                num_workers=dataloader_workers))

    elif val_type == 1:
        m = len(train_data)
        test_data = method(root=data_root,
                           train=False,
                           download=True,
                           transform=transforms_test)
        train_data, val_data = random_split(train_data, [int(m - m * val_account), int(m * val_account)])
        return (data.DataLoader(train_data, batch_size, shuffle=True,
                                num_workers=dataloader_workers),
                data.DataLoader(test_data, batch_size, shuffle=False,
                                num_workers=dataloader_workers),
                data.DataLoader(val_data, batch_size, shuffle=False,
                                num_workers=dataloader_workers),
                )
    elif val_type == 2:
        m = len(train_data)
        train_data, val_data = random_split(train_data, [int(m - m * val_account), int(m * val_account)])
        return data.DataLoader(val_data, batch_size, shuffle=False,
                               num_workers=dataloader_workers)


def get_adv_dataset(attack_dataset_name, attack_method, attack_params, adv_num=200, device=None):
    """
    获取验证的对抗样本-原始数据的数据集
    :param attack_dataset_name:
    :param attack_method:
    :param attack_params:
    :param adv_num:
    :param device:
    :return:
    """
    if attack_dataset_name == "mnist":
        dataset_shape = (adv_num, 1, 28, 28)
        net_type = clfs.ClassiferMNIST
    elif attack_dataset_name == "cifar10":
        dataset_shape = (adv_num, 3, 32, 32)
        net_type = clfs.ResNet18CIFAR10

    else:
        raise Exception("Unknown dataset name")
    net = net_type()
    net.load_exist()
    net.to(device)
    dataset_org = CustomDataset(dataset_shape)
    dataset_adv = CustomDataset(dataset_shape)
    transforms_train = transforms.Compose([transforms.ToTensor()])
    # transforms.Normalize(constants.cifar10_train_mean,
    #                      constants.cifar10_train_std)])

    _, data_iter = load_data(attack_dataset_name, batch_size=1,
                             transforms_train=transforms_train, transforms_test=transforms_train)
    gen_adv_num = 0
    img_num = 0
    if attack_method == "fgsm_i":
        attack = am.fgsm_i
        param_name = 'eps'
    elif attack_method == "df":
        attack = am.deepfool
        param_name = 'none'
    elif attack_method == "cw":
        attack = am.cw
        param_name = 'c'

    else:
        raise Exception("Unknown attack method")
    error_num = 0
    for X, y in data_iter:
        img_num += 1
        X, y = X.to(device), y.to(device)
        h_org = net(X)
        y_org = h_org.argmax(axis=1)
        if y_org != y:
            error_num += 1
            continue
        X_adv, y_adv = attack(net, X, y, device=device)
        if y_adv != y:
            dataset_org.update_data(gen_adv_num, (X.detach().clone().cpu().squeeze(0), y.detach().clone().cpu()))
            dataset_adv.update_data(gen_adv_num,
                                    (X_adv.clone().detach().clone().squeeze(0), y.detach().clone().cpu()))
            gen_adv_num += 1
            print(gen_adv_num)

        if gen_adv_num == adv_num:
            break
    print(f'{error_num}, {img_num}')
    torch.save(dataset_adv,
               f"../data/validation_data/validation_data_{attack_dataset_name}_{attack_method}_adv.pt")
    torch.save(dataset_org,
               f"../data/validation_data/validation_data_{attack_dataset_name}_{attack_method}_org.pt")


def get_cw_dataset(attack_dataset_name, k=0, adv_num=200, device=None):
    """
    获取验证的对抗样本-原始数据的数据集
    :param k:
    :param attack_dataset_name:
    :param attack_method:
    :param attack_params:
    :param adv_num:
    :param device:
    :return:
    """
    if attack_dataset_name == "mnist":
        dataset_shape = (adv_num, 1, 28, 28)
        net_type = clfs.ClassiferMNIST
        transforms_train = transforms.Compose([transforms.ToTensor()])
    elif attack_dataset_name == "cifar10":
        dataset_shape = (adv_num, 3, 32, 32)
        net_type = clfs.ResNet18CIFAR10
        transforms_train = transforms.Compose([transforms.ToTensor(),
                                               transforms.Normalize(constants.cifar10_train_mean,
                                                                    constants.cifar10_train_std)])

    else:
        raise Exception("Unknown dataset name")
    net = net_type()
    net.load_exist()
    net.to(device)
    dataset_org = CustomDataset(dataset_shape)
    dataset_adv = CustomDataset(dataset_shape)

    _, data_iter = load_data(attack_dataset_name, batch_size=1,
                             transforms_train=transforms_train, transforms_test=transforms_train)
    gen_adv_num = 0
    img_num = 0

    error_num = 0
    for X, y in data_iter:
        img_num += 1
        X, y = X.to(device), y.to(device)
        h_org = net(X)
        y_org = h_org.argmax(axis=1)
        if y_org != y:
            error_num += 1
            continue
        X_adv, y_adv = am.cw_l2(net, X, y,k=k, device=device)
        if y_adv != y:
            dataset_org.update_data(gen_adv_num, (X.detach().clone().cpu().squeeze(0), y.detach().clone().cpu()))
            dataset_adv.update_data(gen_adv_num,
                                    (X_adv.clone().detach().clone().squeeze(0), y.detach().clone().cpu()))
            gen_adv_num += 1
            print(gen_adv_num)

        if gen_adv_num == adv_num:
            print(f'{error_num}, {img_num}')
            break

    torch.save(dataset_adv,
               f"../data/validation_data/validation_data_{attack_dataset_name}_{gen_adv_num}_cw_{k}_adv.pt")
    torch.save(dataset_org,
               f"../data/validation_data/validation_data_{attack_dataset_name}__{gen_adv_num}_cw_{k}_org.pt")


if __name__ == "__main__":
    # fgsm_i_attack_params = {
    #     "eps": 0.5,
    #     "alpha": 1,
    #     "iteration": 1000
    # }
    # get_adv_dataset("mnist", "df", adv_num=200, device=try_gpu())
    get_cw_dataset("cifar10", adv_num=200, k=40, device=try_gpu())
