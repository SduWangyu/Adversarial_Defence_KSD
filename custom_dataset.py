import torch
import torchvision
from torch.utils import data
import numpy as np
import classifiers as clfs
import attack_method as am
from torch.utils.data import random_split

dataloader_workers = 4
data_root = "./data"


class CustomDataset(data.Dataset):
    def __init__(self, data_size):
        super(CustomDataset, self).__init__()
        self.x = np.zeros(data_size)
        self.y = np.zeros(data_size[0])

    def __getitem__(self, index):
        img, label = self.x[index], self.y[index]
        return img, label

    def __len__(self):
        return self.x.shape[0]

    def update_data(self, index, data_in):
        self.x[index] = data_in[0]
        self.y[index] = data_in[1]


def load_data(name, batch_size, trans, val_type=0, val_account=0.1):
    """

    :param name: dataset name
    :param batch_size:
    :param trans: transform compose
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
        transform=trans
    )
    if val_type == 0:
        test_data = method(root=data_root,
                           train=False,
                           download=True,
                           transform=trans)

        return (data.DataLoader(train_data, batch_size, shuffle=True,
                                num_workers=dataloader_workers),
                data.DataLoader(test_data, batch_size, shuffle=False,
                                num_workers=dataloader_workers))

    elif val_type == 1:
        m = len(train_data)
        test_data = method(root=data_root,
                           train=False,
                           download=True,
                           transform=trans)
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
    if attack_dataset_name == "mnist":
        dataset_shape = (adv_num, 1, 28, 28)
        net_type = clfs.ClassiferMNIST
    elif attack_dataset_name == "cifar10" or attack_dataset_name == "cifar100":
        dataset_shape = (adv_num, 3, 32, 32)
        net_type = clfs.ClassiferCIFAR10

    else:
        raise Exception("Unknown dataset name")
    net = net_type()
    net.load_exist()
    net.to(device)
    dataset_org = CustomDataset(dataset_shape)
    dataset_adv = CustomDataset(dataset_shape)
    data_iter = load_data(attack_dataset_name, batch_size=1)
    gen_adv_num = 0
    if attack_method == "fgsm_i":
        attack = am.fgsm_i
        param_name = 'eps'
    elif attack_method == "df_t":
        attack = am.deepfool_untarget
        param_name = ' '
    elif attack_method == "df_ut":
        attack = am.deepfool_target
        param_name = ' '
    elif attack_method == "cw":
        attack = am.cw
        param_name = 'c'

    else:
        raise Exception("Unknown attack method")

    for X, y in data_iter:
        X, y = X.to(device), y.to(device)
        X_adv, y_adv = attack(net, X, y, attack_params[param_name], alpha=attack_params['alpha'],
                                 iteration=attack_params['iteration'], device=device)
        if y_adv != y:
            dataset_org.update_data(gen_adv_num, (X.clone().detach().cpu().squeeze(0), y.clone().detach().cpu()))
            dataset_adv.update_data(gen_adv_num,
                                    (X_adv.clone().detach().cpu().squeeze(0), y.clone().detach().cpu()))
            gen_adv_num += 1
            print(gen_adv_num)

        if gen_adv_num == adv_num:
            break

    torch.save(dataset_adv, f"./data/validation_data/validation_data_{attack_dataset_name}_{attack_method}_adv_{param_name}_{attack_params[param_name] * 10}.pt")
    torch.save(dataset_org, f"./data/validation_data/validation_data_{attack_dataset_name}_{attack_method}_org_{param_name}_{attack_params[param_name] * 10}.pt")
