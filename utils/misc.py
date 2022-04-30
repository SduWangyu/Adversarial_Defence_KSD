import torch
from torch.utils import data
import numpy as np
import time
from core import classifiers as clfs, autoencoders as aes, attack_method as am
import deployed_defensive_model as ddm
import utils.evaluate as evaluate


class Timer:
    """记录多次运行时间"""

    def __init__(self):
        """Defined in :numref:`subsec_linear_model`"""
        self.times = []
        self.start()

    def start(self):
        """启动计时器"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并将时间记录在列表中"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间"""
        return np.array(self.times).cumsum().tolist()


class Accumulator:
    """在n个变量上累加"""

    def __init__(self, n):
        """Defined in :numref:`sec_softmax_scratch`"""
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def try_gpu(i=0):
    """如果存在，则返回gpu(i)，否则返回cpu()
    Defined in :numref:`sec_use_gpu`"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def add_noise(img_batch, noise_factor, device):
    """
    给img_batch 添加噪音
    :param device:
    :param img_batch: 要添加的数据
    :param noise_factor: 扰动参数
    :return: img_batch_noise
    """
    return img_batch + (noise_factor * torch.normal(0, 1, img_batch.shape).to(device))


def get_thread_hold_by_prodiv(dataset_name, drop_rate=0.01, p=2, device=None):
    if dataset_name == "mnist":
        autoencoder = aes.ConvAutoEncoderMNIST()
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
        autoencoder = aes.ConvAutoEncoderMNIST()
        autoencoder.load_exist()
        autoencoder.to(device)
        for i in range(1, 6):
            print(i)
            print()
            dataset_adv = torch.load("./data/validation_data/validation_data_mnist_fgsm_adv_eps_0{}.pt".format(i))
            data_iter_adv = data.DataLoader(dataset_adv, batch_size=200)
            l2_distance = []
            for X, _ in data_iter_adv:
                X = X.astype(torch.float32).to(device)
                X_rec = autoencoder(X)
                for x, x_rec in zip(X, X_rec):
                    l2_distance.append(torch.dist(x, x_rec, p=p).cpu().clone().detach())
            l2_distance_tensor = torch.Tensor(l2_distance)
            thr, indices = torch.topk(l2_distance_tensor, 200, dim=0)
            print(thr)

            dataset_adv = torch.load("./data/validation_data/validation_data_mnist_fgsm_org_eps_0{}.pt".format(i))
            data_iter_adv = data.DataLoader(dataset_adv, batch_size=200)
            l2_distance = []
            for X, _ in data_iter_adv:
                X = X.astype(torch.float32).to(device)
                X_rec = autoencoder(X)
                for x, x_rec in zip(X, X_rec):
                    l2_distance.append(torch.dist(x, x_rec, p=p).cpu().clone().detach())
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


def get_test_denfence_dataset(attack_method, dataset_name=None, attack_params=None, val_data_num=100):
    # if dataset_name == "mnist":
    #     dataset = torchvision.datasets.MNIST(
    #         root="./data",
    #         transform=transforms.ToTensor(),
    #         train=True
    #     )
    #     val_data, _ = random_split(dataset, [val_data_num*2, len(dataset)-val_data_num*2])
    #
    # if attack_method == "fgsm":
    #     pass
    #
    # # for X, y in val_data:
    # #
    # torch.save(val_data, "./data/validation_data/validation_data_mnist.pt")
    # attack_method("helloworld", attack_params)
    # dataset = torch.load("./data/validation_data/validation_data_mnist.pt")
    attack_method("helo", attack_params)


def where(cond, x, y):
    """
    code from :
        https://discuss.pytorch.org/t/how-can-i-do-the-operation-the-same-as-np-where/1329/8
    """
    cond = cond.float()
    return (cond * x) + ((1 - cond) * y)


def add_noise(train_data, test_data):
    train_data_noisy = train_data + NOISE_FACTOR * np.random.normal(loc=0.0, scale=1.0, size=train_data.shape)
    test_data_noisy = test_data + NOISE_FACTOR * np.random.normal(loc=0.0, scale=1.0, size=test_data.shape)
    train_data_noisy = np.clip(train_data_noisy, 0., 1.).to(torch.float32)
    test_data_noisy = np.clip(test_data_noisy, 0., 1.).to(torch.float32)
    return train_data_noisy, test_data_noisy

def copy_tensor(x):
    return x.detach().clone()


if __name__ == '__main__':
    # logit_1 = torch.Tensor([[1, 3], [1, 1]])
    # logit_2 = torch.Tensor([[1, 2], [1, 1]])
    # print(jenson_shannon_divergence(logit_1, logit_2))
    # test_defence('mnist', device=try_gpu())
    # print("hello_world")
    # get_test_denfence_dataset(test, attack_params=(1, 2))
    # get_thread_hold_by_distance("mnist", drop_rate=0.01, device=try_gpu())
    get_thread_hold_by_prodiv("mnist", drop_rate=0.01, device=try_gpu())
    # torch.cuda.empty_cache()
    # test_defence('mnist', try_gpu())
    # dataset = torchvision.datasets.MNIST(
    #     root='../data',
    #     train= False,
    #     transform=transforms.ToTensor()
    # )
    # get_thread_hold(dataset, "classifier_mnist", "conv_autoencoder_mnist")
    # a = torch.Tensor((1,2,3))
    # y = torch.Tensor([1, 2])
