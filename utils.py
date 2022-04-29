import torch
import torchvision
from torch import nn
from torch.utils import data
from torchvision import transforms
import numpy as np
import time
import sys
from PIL import Image
import torch.nn.functional as F
import classifiers as clfs
import autoencoders as aes
import attack_method as am
import deployed_defensive_model as ddm

ksd = sys.modules[__name__]


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


def accuracy(y_hat, y):
    """
    计算训练正确的样本的个数
    :param y_hat: 预测值
    :param y: 真实值
    :return:
    """
    # 如果是二维的数据的话，就按行取最大值的索引
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = ksd.argmax(y_hat, axis=1)
    # 按位运算，获取正确的个数
    cmp = ksd.astype(y_hat, y.dtype) == y
    return float(ksd.reduce_sum(ksd.astype(cmp, y.dtype)))


def get_dataloader_workers():
    """使用4个进程来读取数据
    Defined in :numref:`sec_fashion_mnist`"""
    return 4


def load_data_cifar_100(batch_size, resize=None):
    """下载CIFAR100数据集，然后将其加载到内存中
    Defined in :numref:`sec_fashion_mnist`"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    cifar_train = torchvision.datasets.CIFAR100(
        root="../data", train=True, transform=trans, download=True)
    cifar_test = torchvision.datasets.CIFAR100(
        root="../data", train=False, transform=trans, download=True)
    return (data.DataLoader(cifar_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(cifar_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))


def load_data_cifar_10(batch_size, resize=None):
    """下载CIFAR100数据集，然后将其加载到内存中
    Defined in :numref:`sec_fashion_mnist`"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    cifar_train = torchvision.datasets.CIFAR10(
        root="../data", train=True, transform=trans, download=True)
    cifar_test = torchvision.datasets.CIFAR10(
        root="../data", train=False, transform=trans, download=True)
    return (data.DataLoader(cifar_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(cifar_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))


def load_data_mnist(batch_size, resize=None):
    """下载mnist数据集，然后将其加载到内存中
    Defined in :numref:`sec_fashion_mnist`"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.MNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.MNIST(
        root="../data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))


def load_data_mnist_test(batch_size):
    trans = transforms.Compose([transforms.ToTensor()])
    mnist_test = torchvision.datasets.MNIST(
        root="../data", train=False, transform=trans, download=True)

    return data.DataLoader(mnist_test, batch_size, shuffle=False,
                           num_workers=get_dataloader_workers())


def evaluate_accuracy_gpu(net, data_iter, device=None):
    """
    使用device计算精度
    :param net:
    :param data_iter:
    :param device:
    :return:
    """
    if isinstance(net, nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device
    # 正确预测的数量，总预测的数量
    metric = ksd.Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            X = X.to(device)
            y = y.to(device)
            metric.add(ksd.accuracy(net(X), y), ksd.size(y))
    return metric[0] / metric[1]


def evaluate_loss_gpu(net, data_iter, loss_fn, device=None):
    """使用GPU计算自编器在数据集上的精度
    """
    if isinstance(net, nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device

    # loss，总预测的数量
    metric = ksd.Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            X = X.to(device)
            X_R = net(X)
            metric.add(X.shape[0] * loss_fn(X_R, X).item(), ksd.size(y))
    return metric[0] / metric[1]


def train_classifier_and_save(net, train_iter, test_iter, num_epochs, lr, device, model_name='none'):
    """
    使用device 来训练模型
    :param net: 需要训练的网络
    :param train_iter: dataloader迭代器
    :param test_iter: 测试集dataloader迭代器
    :param num_epochs:
    :param lr:  学习率
    :param device:  设备
    :param model_name: 需要保存名字，如果是none的话就不保存
    :return:
    """

    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    # 初始化网络权重
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    # 使用随机梯度下降算法
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    timer, num_batches = ksd.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}\n--------------------------------')
        # 训练损失之和，训练准确率之和，样本数
        metric = ksd.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                # pytorch 计算交叉熵的时候会有一个mean的操作，所以需要乘以batch_size，也就是X.shape[0]
                metric.add(l * X.shape[0], ksd.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                print(f'loss {train_l:.7f}, train acc {(train_acc * 100):>0.2f}%')
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        print(f'test acc {(100 * test_acc):>.2f}%\n')
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')
    if model_name != "none":
        torch.save(net.state_dict(), model_name)

def train_classifier_by_autoencoder_and_save(net, ae,train_iter, test_iter, num_epochs, lr, device, model_name='none'):
    """
    使用device 来训练模型
    :param net: 需要训练的网络
    :param train_iter: dataloader迭代器
    :param test_iter: 测试集dataloader迭代器
    :param num_epochs:
    :param lr:  学习率
    :param device:  设备
    :param model_name: 需要保存名字，如果是none的话就不保存
    :return:
    """

    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    # 初始化网络权重
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    ae.to(device)
    # 使用随机梯度下降算法
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    timer, num_batches = ksd.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}\n--------------------------------')
        # 训练损失之和，训练准确率之和，样本数
        metric = ksd.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                # pytorch 计算交叉熵的时候会有一个mean的操作，所以需要乘以batch_size，也就是X.shape[0]
                metric.add(l * X.shape[0], ksd.accuracy(y_hat, y), X.shape[0])

            optimizer.zero_grad()
            X_rec = ae(X)
            y_hat = net(X_rec)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                # pytorch 计算交叉熵的时候会有一个mean的操作，所以需要乘以batch_size，也就是X.shape[0]
                metric.add(l * X.shape[0], ksd.accuracy(y_hat, y), X.shape[0])

            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                print(f'loss {train_l:.7f}, train acc {(train_acc * 100):>0.2f}%')
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        print(f'test acc {(100 * test_acc):>.2f}%\n')
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')
    if model_name != "none":
        torch.save(net.state_dict(), model_name)


def train_autoencoder(ae, train_iter, test_iter, num_epochs, lr, device):
    """
    用GPU训练自编器
    """

    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    ae.apply(init_weights)
    print('training on', device)
    ae.to(device)
    optimizer = torch.optim.Adam(ae.parameters(), lr=lr)
    loss = nn.MSELoss()
    timer, num_batches = ksd.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        # 训练损失之和，训练准确率之和，样本数
        print(f'Epoch {epoch + 1}\n--------------------------------')
        metric = ksd.Accumulator(2)
        ae.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X = X.to(device)
            X_R = ae(X)
            X = X.reshape(X_R.shape)
            l = loss(X, X_R)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[1]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                print(f'loss {train_l:.7f}')
        test_loss = evaluate_loss_gpu(ae, test_iter, loss)
        print(f'test loss {test_loss:.7f}')
    print(f'loss {train_l:.3f}')
    print(f'{metric[1] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')


def add_noise(img_batch, noise_factor, device):
    """
    给img_batch 添加噪音
    :param img_batch: 要添加的数据
    :param noise_factor: 扰动参数
    :return: img_batch_noise
    """
    return img_batch + (noise_factor * torch.normal(0, 1, img_batch.shape).to(device))


def train_autoencoder_and_save(autoencoder, train_iter, test_iter, num_epochs, lr, device, noise_factor=0.3,
                               model_name="none"):
    """
    用GPU去噪训练自编器
    """

    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    autoencoder.apply(init_weights)
    print('training on', device)
    autoencoder.to(device)
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    timer, num_batches = ksd.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        # 训练损失之和，训练准确率之和，样本数
        print(f'Epoch {epoch + 1}\n--------------------------------')
        metric = ksd.Accumulator(2)
        autoencoder.train()
        for i, (X, _) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X = X.to(device)
            X_noise = add_noise(X, noise_factor, device)
            X_R = autoencoder(X_noise)
            loss = loss_fn(X, X_R)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(loss * X.shape[0], X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[1]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                print(f'loss {train_l:.7f}')
        test_loss = ksd.evaluate_loss_gpu(autoencoder, test_iter, loss_fn)
        print(f'test loss {test_loss:.7f}')
    print(f'{metric[1] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')
    if model_name != "none":
        torch.save(autoencoder.state_dict(), model_name)


class MapIdToHumanString:
    def __init__(self, label_filename=None):
        if not label_filename:
            label_filename = './data/ImageNet/LOC_synset_mapping.txt'
        self.label_lookup = self.load(label_filename)

    def load(self, label_filename):
        label_list = []
        with open(label_filename) as f:
            for line in f:
                label_list.append(line[10:-1:].split(',')[0])

        return label_list.copy()

    def get_label(self, node_id):
        if node_id >= len(self.label_lookup) or node_id < 0:
            return ''
        return self.label_lookup[node_id]


def transform_reverse(ori_image_tensor, totensor_transform):
    """
    reverse
    :param ori_image_tensor:
    :param totensor_transform:
    :return:
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image_tensor = ori_image_tensor.clone().detach().numpy().squeeze().transpose(1, 2, 0)
    if 'Normalize' in str(totensor_transform):
        image_tensor = (image_tensor * std) + mean
    if 'ToTensor' in str(totensor_transform) or image_tensor.max() < 1.0:
        image_tensor = image_tensor * 255.0

    return image_tensor


def transform_to_img_from_dataset(ori_image_tensor, totensor_transform, pic_name):
    """
    从image_tensor到img
    param img_tensor: tensor
    param transforms: torchvision.transforms
    param pic_name: pic path and name
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image_tensor = ori_image_tensor.clone().detach().numpy().squeeze().transpose(1, 2, 0)
    if 'Normalize' in str(totensor_transform):
        image_tensor = (image_tensor * std) + mean
    if 'ToTensor' in str(totensor_transform) or image_tensor.max() < 1.0:
        image_tensor = image_tensor * 255.0

    if image_tensor.shape[2] == 3:
        image = Image.fromarray(image_tensor.astype('uint8')).convert('RGB')
    elif image_tensor.shape[2] == 1:
        image_tensor = image_tensor[0]
        image = Image.fromarray(image_tensor.astype('uint8')).squeeze()
    else:
        raise Exception("Invalid img shape, expected 1 or 3 in axis 2, but got {}!".format(image_tensor.shape[2]))
    image.save(pic_name)
    return image


def image_preprocessing(image_path, image_trasforms, is_unsqueeze=False):
    """
    根据路径对图片进行预处理
    :param is_unsqueeze: 是否加一列
    :param image_path:图片的路径
    :param image_trasforms:torch.transforms
    :return:
    """

    img = Image.open(image_path)
    if is_unsqueeze:
        image_tensor = image_trasforms(img).unsqueeze(0)
    else:
        image_tensor = image_trasforms(img)
    return image_tensor


def show_images_diff(original_img_tensor, adversarial_img_tensor, ToTensor_TransForms, original_label, adversial_label):
    """
    对比展现原始图片和对抗样本图片
    :param original_label:
    :param ToTensor_TransForms:
    :param original_img:
    :param adversarial_img:
    :return:
    """
    map_id_to_label_string = MapIdToHumanString()
    import matplotlib.pyplot as plt
    original_img = transform_reverse(original_img_tensor, ToTensor_TransForms)
    adversarial_img = transform_reverse(adversarial_img_tensor, ToTensor_TransForms)
    original_img = np.clip(original_img, 0, 255).astype(np.uint8)
    adversarial_img = np.clip(adversarial_img, 0, 255).astype(np.uint8)
    if original_img.any() > 1.0:
        original_img = original_img / 255.0
    if adversarial_img.any() > 1.0:
        adversarial_img = adversarial_img / 255.0
    plt.figure()

    plt.subplot(131)
    plt.title('Original')
    plt.title('lable:{}'.format(map_id_to_label_string.get_label(original_label)), y=-0.16, loc="right")
    plt.imshow(original_img)
    plt.axis('off')

    plt.subplot(132)
    plt.title('Adversarial')
    plt.imshow(adversarial_img)
    plt.title('lable:{}'.format(map_id_to_label_string.get_label(adversial_label)), y=-0.16, loc="right")
    plt.axis('off')

    plt.subplot(133)
    plt.title('Adversarial-Original')
    difference = adversarial_img - original_img
    # (-1,1)  -> (0,1)
    difference = difference / abs(difference).max() / 2.0 + 0.5
    plt.imshow(difference, cmap=plt.cm.gray)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def jenson_shannon_divergence(net_1_logits, net_2_logits):
    from torch.functional import F
    net_1_probs = F.softmax(net_1_logits, dim=1)
    net_2_probs = F.softmax(net_2_logits, dim=1)

    total_m = 0.5 * (net_1_probs + net_2_probs)
    loss = 0.0
    loss += F.kl_div(F.log_softmax(net_1_logits, dim=1), total_m, reduction="none")
    loss += F.kl_div(F.log_softmax(net_2_logits, dim=1), total_m, reduction="none")
    return (0.5 * ksd.reduce_sum(loss, 1))


def get_thread_hold(dataset, classifier_name, autoencoder_name, drop_rate=0.01, device=try_gpu()) -> torch.Tensor:
    """
    通过dataset来获取拒绝的thread_hold，drop_rate是可能的误报率
    :param classifier: 分类器
    :param autoencoder: 用的autoencoder
    :param dataset: 用于那个的dataset
    :param drop_rate: 误报率
    :return:
    MNIST 2.5337
    """
    classsifier_file_path = './models/classifiers/'
    pth = '.pth'
    autoencoder_file_path = './models/autoencoders/'
    dataloader = data.DataLoader(dataset, batch_size=128)
    th_idx = int(drop_rate * len(dataset))
    if classifier_name == 'classifier_mnist':
        clf = clfs.Classifer_MNIST()
        clf.load_state_dict(torch.load(classsifier_file_path + classifier_name + pth))
        clf.eval().to(device)

    if autoencoder_name == 'conv_autoencoder_mnist':
        ae = aes.ConvAutoEncoder(encoded_space_dim=10, fc2_input_dim=128)
        ae.load_state_dict(torch.load(autoencoder_file_path + autoencoder_name + pth))
        ae.eval().to(device)

    jsd_tensor = torch.Tensor().to(device)
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            org_pre = clf(X)  # 原本的概率
            rec_pre = clf(ae(X))  # 通过ae后的概率
            jsd = jenson_shannon_divergence(org_pre, rec_pre)
            jsd_tensor = torch.cat((jsd_tensor, jsd), 0)

    thr, indices = torch.topk(jsd_tensor, th_idx, dim=0)
    print(thr)
    return thr[-1]


def test_defence(dataset_name, device=None, attack_name='fgsm'):
    if dataset_name == 'mnist':
        net = ddm.Defence_MNIST()
        net.to(device)
        net.eval()
        test_num = 100
        test_data_set = ksd.load_data_mnist_test(test_num)
        find_1 = 0
        fix_1 = 0
        totall_adv = 0
        totall_norm = 0
        error_num = 0
        e = 0.1
        max_iterations = 1000
        for (X_iter, y_iter) in test_data_set:
            X_iter, y_iter = X_iter.to(device), y_iter.to(device)
            for i in range(100):
                if i % 10 == 0:
                    print(f'{i + 1} --------')
                x_adv = am.fgsm(net.classifier, X_iter[i], e=e, max_iterations=max_iterations,
                                target_label=(y_iter[i] + 1) % 10, device=device)
                org_pre, is_adv = net(torch.unsqueeze(X_iter[i], 0))
                if is_adv:
                    error_num += 1
                if x_adv is not None:
                    with torch.no_grad():
                        adv_pre, is_adv = net(torch.unsqueeze(x_adv, 0))
                        totall_adv += 1
                        if ksd.argmax(adv_pre, 1) == y_iter[i]:
                            fix_1 += 1
                        if is_adv:
                            find_1 += 1
            print(f'{attack_name}, e={e}, max_epochs={max_iterations}')
            print(f'共有对抗样本：{totall_adv}, 发现对抗样本：{find_1}，发现率：{100 * find_1 / totall_adv:.2f}%')
            print(f'修复：{fix_1}, 修复率：{100 * fix_1 / totall_adv:.2f}%')
            print(f'共有正常：{100}, 发现对抗样本：{error_num}，误报率率：{100 * error_num / 100:.2f}%')
            break


def show_images_ae_minst(decode_images, x_test):
    """
    plot the images.
    :param decode_images: the images after decoding
    :param x_test: testing data
    :return:
    """
    from matplotlib import pyplot as plt
    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(n):
        ax = plt.subplot(2, n, i+1)
        ax.imshow(x_test[i].reshape(28, 28, 1))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax = plt.subplot(2, n, i + 1 + n)
        ax.imshow(decode_images[i].detach().numpy().reshape(28, 28, 1))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


def show_images_ae_cifar10(decode_images, x_test):
    """
    plot the images.
    :param decode_images: the images after decoding
    :param x_test: testing data
    :return:
    """
    from matplotlib import pyplot as plt
    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(n):
        ax = plt.subplot(2, n, i+1)
        x_test_show = x_test[i].transpose(1, 0).transpose(2, 1)
        ax.imshow(x_test_show)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax = plt.subplot(2, n, i + 1 + n)
        ax.imshow(decode_images[i].detach().numpy().transpose(1, 2, 0))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

def where(cond, x, y):
    """
    code from :
        https://discuss.pytorch.org/t/how-can-i-do-the-operation-the-same-as-np-where/1329/8
    """
    cond = cond.float()
    return (cond*x) + ((1-cond)*y)

argmax = lambda x, *args, **kwargs: x.argmax(*args, **kwargs)
astype = lambda x, *args, **kwargs: x.type(*args, **kwargs)
reduce_sum = lambda x, *args, **kwargs: x.sum(*args, **kwargs)
size = lambda x, *args, **kwargs: x.numel(*args, **kwargs)

if __name__ == '__main__':
    # logit_1 = torch.Tensor([[1, 3], [1, 1]])
    # logit_2 = torch.Tensor([[1, 2], [1, 1]])
    # print(jenson_shannon_divergence(logit_1, logit_2))
    test_defence('mnist', device=try_gpu())
    # print("hello_world")
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
