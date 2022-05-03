import torch
from torchvision.transforms import ToTensor, Compose
import cv2
import utils.misc
import utils.misc as misc
from core.custom_dataset import *
from core.autoencoders import *
from core.deployed_defensive_model import *
from utils import evaluate


def get_thread_hold_by_prodiv(dataset_name, attack_method, drop_rate=0.01, p=2, device=None):
    if dataset_name == "mnist":
        autoencoder = ConvAutoEncoderMNIST()
        autoencoder.load_exist(path="../data/models_trained/autoencoders/cov_mnist.pth")
        autoencoder.to(device)
        classifier = clfs.ClassiferMNIST()
        classifier.load_exist()
        classifier.to(device)
    if dataset_name == "cifar10":
        autoencoder = ConvAutoEncoderCIFAR10()
        autoencoder.load_exist(path="../data/models_trained/autoencoders/cov_cifar10_colab_noise_factor_001.pth")
        autoencoder.to(device)
        classifier = clfs.ResNet18CIFAR10()
        classifier.load_exist(path="../data/models_trained/classifiers/resnet18_cifar10.pth")
        classifier.to(device)

    i_i = range(0, 50, 10)
    for i in i_i:
        print(i)
        dataset_adv = torch.load(f"../data/validation_data/validation_data_{dataset_name}_{attack_method}_{i}_adv.pt")
        data_iter_adv = data.DataLoader(dataset_adv, batch_size=200)
        for X, _ in data_iter_adv:
            X = X.type(torch.float32).to(device)
            X_rec = autoencoder(X)
            jsd_tensor = evaluate.jenson_shannon_divergence(classifier(X), classifier(X_rec))

        thr, indices = torch.topk(jsd_tensor, 200, dim=0)
        print(thr)

        dataset_adv = torch.load(f"../data/validation_data/validation_data_{dataset_name}_{attack_method}_{i}_org.pt")
        data_iter_adv = data.DataLoader(dataset_adv, batch_size=200)
        for X, _ in data_iter_adv:
            X = X.type(torch.float32).to(device)
            X_rec = autoencoder(X)
            jsd_tensor = evaluate.jenson_shannon_divergence(classifier(X), classifier(X_rec))

        thr, indices = torch.topk(jsd_tensor, 200, dim=0)
        print(thr)


def test_defence_by_distance(dataset_name, attack_method, drop_rate=0.01, p=2, device=None):
    if dataset_name == "mnist":
        autoencoder = ConvAutoEncoderMNIST()
        autoencoder.load_exist(path="../data/models_trained/autoencoders/cov_mnist.pth")
        autoencoder.to(device)
    elif dataset_name == "cifar10":
        autoencoder = ConvAutoEncoderCIFAR10()
        autoencoder.load_exist(path="../data/models_trained/autoencoders/cov_cifar10_colab_noise_factor_08.pth")
        autoencoder.to(device)
    else:
        raise Exception("No such attack method.")
    # i_i = [1, 5, 10]
    i_i = range(0, 50, 10)
    for i in i_i:
        print(i)
        print()
        dataset_org = torch.load(f"../data/validation_data/validation_data_{dataset_name}_{attack_method}_{i}_org.pt")
        data_iter_org = DataLoader(dataset_org, batch_size=200)
        l2_distance = []
        for X, _ in data_iter_org:
            X = X.type(torch.float32).to(device)
            X_rec = autoencoder(X)
            for x, x_rec in zip(X, X_rec):
                l2_distance.append(torch.dist(x, x_rec, p=p).cpu().detach().clone())
            break
        l2_distance_tensor = torch.Tensor(l2_distance)
        thr, indices = torch.topk(l2_distance_tensor, 200, dim=0)
        print(thr)

        dataset_adv = torch.load(f"../data/validation_data/validation_data_{dataset_name}_{attack_method}_{i}_adv.pt")
        data_iter_adv = DataLoader(dataset_adv, batch_size=200)
        # data_iter_org = load_data("cifar10", batch_size=15)
        l2_distance = []
        for X, _ in data_iter_adv:
            X = X.type(torch.float32).to(device)
            X_rec = autoencoder(X)
            for x, x_rec in zip(X, X_rec):
                l2_distance.append(torch.dist(x, x_rec, p=p).cpu().clone().detach())
            break
        l2_distance_tensor = torch.Tensor(l2_distance)
        thr, indices = torch.topk(l2_distance_tensor, 200, dim=0)
        print(thr)

        # dataset_org = torch.load("./data/validation_data/validation_data_mnist_fgsm_org_eps_0{}.pt".format(i))
        # data_iter_org = data.DataLoader(dataset_adv, batch_size=200)


# def test_defence(dataset_name, device=None, attack_name='fgsm'):
#     if dataset_name == 'mnist':
#         net = ddm.DefenceMNIST()
#         net.to(device)
#         net.eval()
#         test_num = 100
#         test_data_set = load_data_mnist_test(test_num)
#         find_1 = 0
#         fix_1 = 0
#         totall_adv = 0
#         totall_norm = 0
#         error_num = 0
#         e = 0.1
#         max_iterations = 1000
#         for (X_iter, y_iter) in test_data_set:
#             X_iter, y_iter = X_iter.to(device), y_iter.to(device)
#             for i in range(100):
#                 if i % 10 == 0:
#                     print(f'{i + 1} --------')
#                 x_adv = am.fgsm(net.classifier, X_iter[i], e=e, max_iterations=max_iterations,
#                                 target_label=(y_iter[i] + 1) % 10, device=device)
#                 org_pre, is_adv = net(torch.unsqueeze(X_iter[i], 0))
#                 if is_adv:
#                     error_num += 1
#                 if x_adv is not None:
#                     with torch.no_grad():
#                         adv_pre, is_adv = net(torch.unsqueeze(x_adv, 0))
#                         totall_adv += 1
#                         if adv_pre.argmax(1) == y_iter[i]:
#                             fix_1 += 1
#                         if is_adv:
#                             find_1 += 1
#             print(f'{attack_name}, e={e}, max_epochs={max_iterations}')
#             print(f'共有对抗样本：{totall_adv}, 发现对抗样本：{find_1}，发现率：{100 * find_1 / totall_adv:.2f}%')
#             print(f'修复：{fix_1}, 修复率：{100 * fix_1 / totall_adv:.2f}%')
#             print(f'共有正常：{100}, 发现对抗样本：{error_num}，误报率率：{100 * error_num / 100:.2f}%')
#             break


def test_denfence_dataset(dataset_name, attack_method, device="cpu"):
    net = DefenceMNIST()
    net.to(device)
    i_i = range(0, 50, 10)
    for i in i_i:
        data_adv = torch.load(f"../data/validation_data/validation_data_{dataset_name}_{attack_method}_{i}_adv.pt")
        data_org = torch.load(f"../data/validation_data/validation_data_{dataset_name}_{attack_method}_{i}_org.pt")
        data_iter_adv = DataLoader(data_adv, batch_size=1)
        data_iter_org = DataLoader(data_org, batch_size=1)
        yep = 0
        fix = 0
        fix_1 = 0
        error = 0
        for (x_adv, y_adv), (x_org, y_org) in zip(data_iter_adv, data_iter_org):
            x_adv, y_adv = x_adv.type(torch.float).to(device), y_adv.to(device)
            y_pre, y_is_adv = net(x_adv)
            if y_is_adv:
                yep += 1
            else:
                y_pre.argmax(1) == y_adv
                fix += 1
            x_org, y_org = x_org.type(torch.float).to(device), y_org.to(device)
            y_pre, y_is_adv = net(x_org)
            if y_is_adv:
                error += 1
            else:
                y_pre.argmax(1) == y_org
                fix_1 += 1
        print('查出的对抗样本数量：', yep)
        print('误报的对抗样本数量：', error)
        print('修复的对抗样本数量：', fix)
        print('正常样本的识别准确率：', fix_1)
        print()


def get_thread_hold_by_distance(dataset_name, drop_rate=0.01, p=1, device=None):
    if dataset_name == "mnist":
        autoencoder = ConvAutoEncoderMNIST()
        autoencoder.load_exist(path="../data/models_trained/autoencoders/cov_mnist.pth")
        autoencoder.to(device)
    if dataset_name == "cifar10":
        autoencoder = ConvAutoEncoderCIFAR10()
        autoencoder.load_exist(path="../data/models_trained/autoencoders/cov_cifar10_colab_noise_factor_001.pth")
        autoencoder.to(device)
    validation_data_iter = load_data(dataset_name, val_type=2, val_account=0.1, batch_size=1,
                                     transforms_train=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]),
                                     transforms_test=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))
    lp_distance = []
    for X, _ in validation_data_iter:
        X = X.to(device)
        X_r = autoencoder(X)
        lp_distance.append(torch.dist(X, X_r, p=p).cpu().detach().clone())
    print(len(lp_distance))
    lp_distance = torch.Tensor(lp_distance)
    thr, indices = torch.topk(lp_distance, int(drop_rate * lp_distance.shape[0]), dim=0)
    print(thr)
    return


def denoising_by_cv2(org_image):
    tmp_image = misc.copy_tensor(org_image).cpu().numpy().transpose(1, 2, 0)
    tmp_image = cv2.medianBlur(tmp_image, 3)


    res_image = torch.Tensor()
    return res_image


if __name__ == "__main__":
    # get_thread_hold_by_distance"cifar10", "fgsm_i", device=misc.try_gpu())
    # test_defence_by_distance("cifar10", "cw", p=1, device=misc.try_gpu())
    # get_thread_hold_by_prodiv("cifar10", "cw", device=misc.try_gpu())
    denoising_by_cv2(torch.rand())
    test_denfence_dataset("mnist", "cw", device=misc.try_gpu())
