from torch import nn

import torch
from utils.misc import Timer, Accumulator, copy_tensor
import utils.evaluate as evaluate


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)


def add_noise(img_tensor, noise_factor):
    noised_img_tensor = copy_tensor(img_tensor)
    noised_img_tensor.data += noise_factor * torch.normal(0, 1, size=img_tensor.data.shape).to(img_tensor.data.device)
    noised_img_tensor = torch.clamp(noised_img_tensor, 0., 1.)
    return noised_img_tensor


def train_classifier(net, train_iter, test_iter, num_epochs, lr, device, model_name=None):
    """
    使用device 来训练模型
    :param ae: 传递自编器，若为空则不使用
    :param net: 需要训练的网络
    :param train_iter: dataloader迭代器
    :param test_iter: 测试集dataloader迭代器
    :param num_epochs:
    :param lr:  学习率
    :param device:  设备
    :param model_name: 需要保存名字，若不传递值的话则不保存
    :return:
    """

    # 初始化网络权重
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    best_acc = 0

    # 使用随机梯度下降算法
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    train_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
    loss = nn.CrossEntropyLoss()
    timer, num_batches = Timer(), len(train_iter)

    max_test_acc = 0
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}\n--------------------------------')
        # 训练损失之和，训练准确率之和，样本数
        metric = Accumulator(3)
        net.train()
        if epoch > 0:
            train_scheduler.step()
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
                metric.add(l * X.shape[0], evaluate.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                print(f'loss {train_l:.7f}, train acc {(train_acc * 100):>0.2f}%')

        test_acc = evaluate.accuracy_gpu(net, test_iter)
        if test_acc > best_acc:
            if model_name:
                torch.save(net.state_dict(), model_name)
            best_acc = test_acc
        print(f'test acc {(100 * test_acc):>.2f}% ,best acc{(100 * best_acc):>.2f}% \n')
        scheduler.step()

    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')


def train_autoencoder(autoencoder, train_iter, test_iter, num_epochs, lr, device, noise_factor=0.1,
                      model_name=None):
    """
    用GPU去噪训练自编器
    """
    autoencoder.apply(init_weights)
    print('training on', device)
    autoencoder.to(device)
    optimizer = torch.optim.Adam(autoencoder.parameters())
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    loss_fn = nn.MSELoss()
    num_batches = len(train_iter)
    best_loss = 999
    for epoch in range(num_epochs):
        # 训练损失之和，训练准确率之和，样本数
        print(f'Epoch {epoch + 1}\n--------------------------------')
        metric = Accumulator(2)
        autoencoder.train()
        for i, (X, _) in enumerate(train_iter):
            optimizer.zero_grad()
            X = X.to(device)
            X_noise = add_noise(X, noise_factor)
            X_R = autoencoder(X_noise)
            loss = loss_fn(X, X_R)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(loss * X.shape[0], X.shape[0])
            train_l = metric[0] / metric[1]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                print(f'loss {train_l:.7f}')
        test_loss = evaluate.loss_gpu(autoencoder, test_iter, loss_fn)
        print(f'test loss {test_loss:.7f}')
        # scheduler.step()

        if best_loss > test_loss:
            best_loss = test_loss
            if model_name:
                torch.save(autoencoder.state_dict(), model_name)
