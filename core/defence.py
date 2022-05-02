import torch
from torchvision.transforms import ToTensor, Compose
import utils.misc as misc
from core.custom_dataset import *
from core.autoencoders import *
from core.deployed_defensive_model import *
def get_thread_hold_by_prodiv(dataset_name, drop_rate=0.01, p=2, device=None):
    if dataset_name == "mnist":
        autoencoder = ConvAutoEncoderMNIST()
        autoencoder.load_exist()
        autoencoder.to(device)
        classifier = clfs.ClassiferMNIST()
        classifier.load_exist()
        classifier.to(device)
        for i in range(1, 6):
            print(i)
            dataset_adv = torch.load("./data/validation_data/validation_data_mnist_fgsm_adv_eps_0{}.pt".format(i))
            data_iter_adv = data.DataLoader(dataset_adv, batch_size=200)
            for X, _ in data_iter_adv:
                X = X.astype(torch.float32).to(device)
                X_rec = autoencoder(X)
                jsd_tensor = evaluate.jenson_shannon_divergence(classifier(X), classifier(X_rec))

            thr, indices = torch.topk(jsd_tensor, 200, dim=0)
            print(thr)

            dataset_adv = torch.load("./data/validation_data/validation_data_mnist_fgsm_org_eps_0{}.pt".format(i))
            data_iter_adv = data.DataLoader(dataset_adv, batch_size=200)
            for X, _ in data_iter_adv:
                X = X.astype(torch.float32).to(device)
                X_rec = autoencoder(X)
                jsd_tensor = evaluate.jenson_shannon_divergence(classifier(X), classifier(X_rec))

            thr, indices = torch.topk(jsd_tensor, 200, dim=0)
            print(thr)


def get_thread_hold_by_distance(dataset_name, drop_rate=0.01, p=2, device=None):
    if dataset_name == "mnist":
        autoencoder = ConvAutoEncoderCIFAR10()
        autoencoder.load_exist(path="../data/models_trained/autoencoders/cov_cifar100.pth")
        autoencoder.to(device)
    if dataset_name == "cifar10":
        autoencoder = ConvAutoEncoderCIFAR10()
        autoencoder.load_exist(path="../data/models_trained/autoencoders/cov_cifar100.pth")
        autoencoder.to(device)
    i_i = [1]
    data_iter_adv, _ = load_data("cifar10", batch_size=15, transforms_train=Compose([ToTensor()]),
                                 transforms_test=Compose([ToTensor()]))
    for i in i_i:
        print(i)
        if i == 30:
            continue
        print()
        dataset_adv = torch.load("../data/validation_data/validation_data_cifar10_fgsm_i_{}.0_adv.pt".format(i))

        l2_distance = []
        for X, _ in data_iter_adv:
            X = X.type(torch.float32).to(device)
            X_rec = autoencoder(X)
            for x, x_rec in zip(X, X_rec):
                l2_distance.append(torch.dist(x, x_rec, p=p).cpu().detach().clone())
            break
        l2_distance_tensor = torch.Tensor(l2_distance)
        print(l2_distance_tensor.shape)
        thr, indices = torch.topk(l2_distance_tensor, 15, dim=0)
        print(thr)
        # print(l2_distance_tensor)
        # dataset_adv = torch.load("../data/validation_data/validation_data_cifar10_fgsm_i_{}.0_org.pt".format(i))
        # data_iter_adv = DataLoader(dataset_adv, batch_size=200)
        # data_iter_org = load_data("cifar10", batch_size=15)
        # l2_distance = []
        # for X, _ in data_iter_adv:
        #     X = X.type(torch.float32).to(device)
        #     X_rec = autoencoder(X)
        #     for x, x_rec in zip(X, X_rec):
        #         l2_distance.append(torch.dist(x, x_rec, p=p).cpu().clone().detach())
        # l2_distance_tensor = torch.Tensor(l2_distance)
        # thr, indices = torch.topk(l2_distance_tensor, 200, dim=0)
        # print(thr)

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


def test_denfence_dataset_mnist(attack_method,device="cpu"):
    net = DefenceMNIST()
    net.to(device)
    if attack_method == "fgsm":
        for i in range(0, 50, 10):
            import os
            print(os.getcwd())
            data_adv = torch.load("../data/validation_data/validation_data_mnist_200_cw_{}_adv.pt".format(i))
            data_org = torch.load("../data/validation_data/validation_data_mnist__200_cw_{}_org.pt".format(i))
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
                y_pre, y_is_adv = net(x_adv)
                if y_is_adv:
                    error += 1
                else:
                    y_pre.argmax(1) == y_org
                    fix_1 += 1
            print(yep)
            print(error)
            print(fix)
            print(fix_1)
            print()





if __name__ == "__main__":
    # get_thread_hold_by_distance("cifar10", device=misc.try_gpu())

    test_denfence_dataset_mnist("fgsm", device=misc.try_gpu())