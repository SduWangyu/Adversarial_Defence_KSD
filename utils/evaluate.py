from torch import nn

from utils.utils import *


def accuracy(y_hat, y):
    """
    计算训练正确的样本的个数
    :param y_hat: 预测值
    :param y: 真实值
    :return:
    """
    # 如果是二维的数据的话，就按行取最大值的索引
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = argmax(y_hat, axis=1)
    # 按位运算，获取正确的个数
    cmp = astype(y_hat, y.dtype) == y
    return float(reduce_sum(astype(cmp, y.dtype)))


def accuracy_gpu(net, data_iter, ae=None, device=None):
    """
    使用device计算精度
    :param ae:
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
    metric = Accumulator(2)
    with torch.no_grad():
        if ae:
            for X, y in data_iter:
                X = X.to(device)
                X_rec = ae(X)
                y = y.to(device)
                metric.add(accuracy(net(X), y), size(y))
                metric.add(accuracy(net(X_rec), y), size(y))
        else:
            for X, y in data_iter:
                X = X.to(device)
                y = y.to(device)
                metric.add(accuracy(net(X), y), size(y))
    return metric[0] / metric[1]


def loss_gpu(net, data_iter, loss_fn, device=None):
    """使用GPU计算自编器在数据集上的精度
    """
    if isinstance(net, nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device

    # loss，总预测的数量
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            X = X.to(device)
            X_R = net(X)
            metric.add(X.shape[0] * loss_fn(X_R, X).item(), size(y))
    return metric[0] / metric[1]


def jenson_shannon_divergence(net_1_logit, net_2_logit):
    from torch.functional import F
    net_1_probs = F.softmax(net_1_logit, dim=1)
    net_2_probs = F.softmax(net_2_logit, dim=1)

    total_m = 0.5 * (net_1_probs + net_2_probs)
    loss = 0.0
    loss += F.kl_div(F.log_softmax(net_1_logit, dim=1), total_m, reduction="none")
    loss += F.kl_div(F.log_softmax(net_2_logit, dim=1), total_m, reduction="none")
    return 0.5 * reduce_sum(loss, 1)
