import threading

import torch
import torchvision
from torch.utils import data
import utils.constants
import core.attack_method as am
from torch.utils.data import random_split
import core.classifiers as clfs
from utils import constants
from utils.misc import try_gpu
from torchvision import transforms
from torch.utils.data import DataLoader

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


class MyDataset(data.Dataset):
    def __init__(self, data_size):
        super(MyDataset, self).__init__()
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


def get_fgsm_dataset(attack_dataset_name, eps, adv_num=200, device=None):
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
        transforms_train = transforms.Compose([transforms.ToTensor()])
    elif attack_dataset_name == "cifar10":
        dataset_shape = (adv_num, 3, 32, 32)
        net_type = clfs.ResNet18CIFAR10
        transforms_train = transforms.Compose([transforms.ToTensor(),
                                               transforms.Normalize(utils.constants.cifar10_test_mean,
                                                                    utils.constants.cifar10_train_std)])

    else:
        raise Exception("Unknown dataset name")
    net = net_type()
    net.load_exist()
    net.to(device)
    dataset_org = CustomDataset(dataset_shape)
    dataset_adv = CustomDataset(dataset_shape)

    # transforms.Normalize(constants.cifar10_train_mean,
    #                      constants.cifar10_train_std)])

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
        X_adv, y_adv = am.fgsm_i(net, X, y, eps=eps, device=device)
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
               f"../data/validation_data/validation_data_{attack_dataset_name}_fgsm_i_{eps * 100}_adv.pt")
    torch.save(dataset_org,
               f"../data/validation_data/validation_data_{attack_dataset_name}_fgsm_i_{eps * 100}_org.pt")


global_total_adv_num = 0
global_target_adv_num = 0


class CWThread(threading.Thread):
    adv_num_lock = threading.Lock()

    def __init__(self, func, args=()):
        super(CWThread, self).__init__()
        self.result = None
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        threading.Thread.join(self)
        return self.result


def cw_process(name, net, data_iter, k, device, subset_shape):
    global global_total_adv_num, global_target_adv_num
    now_adv_num = 0
    dataset_org = CustomDataset(subset_shape)
    dataset_adv = CustomDataset(subset_shape)
    img_num = 0
    error_num = 0

    for X, y in data_iter:
        # print(f'[{name}]\t{y}')
        y = y.long()
        img_num += 1
        if not img_num % 100:
            print(f'[{name}]\timg_num: {img_num}')

        X, y = X.to(device), y.to(device)
        h_org = net(X)
        y_org = h_org.argmax(axis=1)
        # print(f'[{name}]\t{y}, {y_org}')
        if y_org != y:
            error_num += 1
            # print(f'[{name}]\t{error_num}')
            continue
        X_adv, y_adv = am.cw_l2(net, X, y, k=k, device=device)
        # print(f'[{name}]\t{y}, {y_org}, {y_adv}')
        if global_target_adv_num == global_total_adv_num:
            print(f'[{name}]\t{error_num}, {img_num}')
            break
        CWThread.adv_num_lock.acquire()
        if y_adv != y and global_target_adv_num > global_total_adv_num:
            dataset_org.update_data(now_adv_num,
                                    (X.detach().clone().cpu().squeeze(0), y.detach().clone().cpu()))
            dataset_adv.update_data(now_adv_num,
                                    (X_adv.clone().detach().clone().squeeze(0), y.detach().clone().cpu()))
            now_adv_num += 1
            global_total_adv_num += 1
            print(f'[{name}]\tGot {now_adv_num}')
        CWThread.adv_num_lock.release()
        if global_target_adv_num == global_total_adv_num:
            print(f'[{name}]\t{error_num}, {img_num}')
            break
    print(f'[{name}]\tDone.')
    return dataset_adv, dataset_org, error_num, img_num, now_adv_num


def get_cw_dataset(attack_dataset_name, k=0, target_adv_num=200, device=None):
    """
    获取验证的对抗样本-原始数据的数据集
    :param target_adv_num:
    :param attack_dataset_name:
    :param k:
    :param device:
    :return:
    """
    total_error_num = 0
    total_img_num = 0
    global global_target_adv_num
    global_target_adv_num = target_adv_num
    if attack_dataset_name == "mnist":
        dataset_shape = (1, 28, 28)
        net_type = clfs.ClassiferMNIST
        transforms_train = transforms.Compose([transforms.ToTensor()])
    elif attack_dataset_name == "cifar10":
        dataset_shape = (3, 32, 32)
        net_type = clfs.ResNet18CIFAR10
        transforms_train = transforms.Compose([transforms.ToTensor(),
                                               transforms.Normalize(constants.cifar10_train_mean,
                                                                    constants.cifar10_train_std)])
    else:
        raise Exception("Unknown dataset name")

    dataset_org = CustomDataset((target_adv_num,) + dataset_shape)
    dataset_adv = CustomDataset((target_adv_num,) + dataset_shape)

    net = net_type()
    net.load_exist()
    net.to(device)

    _, data_iter = load_data(attack_dataset_name, batch_size=1,
                             transforms_train=transforms_train, transforms_test=transforms_train)

    import numpy as np
    array = np.array(range(data_iter.dataset.data.shape[0]))
    np.random.shuffle(array)

    gen_threads = []
    tmp_num = 0
    for _ in range(constants.thread_num):
        tmp_dataset = CustomDataset((len(data_iter.dataset) // constants.thread_num, ) + dataset_shape)

        for i in range(len(data_iter.dataset) // constants.thread_num):
            tmp_dataset.update_data(i, data_iter.dataset[tmp_num])
            tmp_num += 1

        tmp_dataset = DataLoader(tmp_dataset, batch_size=1)
        gen_thread = CWThread(cw_process, (str(_), net, tmp_dataset, k, device, (target_adv_num,) + dataset_shape))
        # cw_process(str(_), net, tmp_dataset, k, device, (target_adv_num // constants.thread_num,) + dataset_shape)
        gen_threads.append(gen_thread)
        gen_thread.start()

    tmp_num = 0
    for res in gen_threads:
        tmp_dataset_adv, tmp_dataset_org, tmp_error_num, tmp_img_num, tmp_adv_num = res.get_result()
        total_error_num += tmp_error_num
        total_img_num += tmp_img_num
        for i in range(tmp_adv_num):
            dataset_adv.update_data(tmp_num, tmp_dataset_adv[i])
            dataset_org.update_data(tmp_num, tmp_dataset_org[i])
            tmp_num += 1
            if tmp_num == target_adv_num:
                break
        else:
            continue
        break
    print(f'{total_img_num}, {total_error_num}')
    torch.save(dataset_adv,
               f"../data/validation_data/validation_data_{attack_dataset_name}_cw_{k}_adv.pt")
    torch.save(dataset_org,
               f"../data/validation_data/validation_data_{attack_dataset_name}_cw_{k}_org.pt")


def get_cw_dataset_test(attack_dataset_name, k=0, adv_num=200, device=None):
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
        print(y)
        exit(0)
        X, y = X.to(device), y.to(device)
        h_org = net(X)
        y_org = h_org.argmax(axis=1)
        if y_org != y:
            error_num += 1
            continue
        X_adv, y_adv = am.cw_l2(net, X, y, k=k, device=device)
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
    get_cw_dataset("cifar10", k=30, target_adv_num=200, device=try_gpu())
