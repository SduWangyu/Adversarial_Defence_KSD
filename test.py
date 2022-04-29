# import matplotlib.pyplot as plt
# # plotting library
# import numpy as np
# # this module is useful to work with numerical arrays
# import pandas as pd
# import random
# import torch
# import torchvision
# from torchvision import transforms
# from torch.utils.data import DataLoader, random_split
# from torch import nn
# import torch.nn.functional as F
# import torch.optim as optim
# from autoencoders import ConvAutoEncoder
# import utils as ksd
#
#
# data_dir = 'dataset'
# train_dataset = torchvision.datasets.MNIST(data_dir, train=True, download=True)
# test_dataset = torchvision.datasets.MNIST(data_dir, train=False, download=True)
# train_transform = transforms.Compose([transforms.ToTensor(), ])
# test_transform = transforms.Compose([transforms.ToTensor(), ])
# train_dataset.transform = train_transform
# test_dataset.transform = test_transform
# m = len(train_dataset)
# train_data, val_data = random_split(train_dataset, [int(m - m * 0.2), int(m * 0.2)])
# batch_size = 64
# train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
# valid_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
#
#
#
#
# def add_noise(img_batch, noise_factor, device):
#     """
#     给img_batch 添加噪音
#     :param img_batch: 要添加的数据
#     :param noise_factor: 扰动参数
#     :return: img_batch_noise
#     """
#     return img_batch + (noise_factor * torch.normal(0, 1, img_batch.shape)).to(device)
#
#
# def train_autoencoder_and_save(autoencoder, train_iter, test_iter, num_epochs, lr, device, noise_factor=0.3, model_name="none"):
#     """
#     用GPU训练自编器
#     """
#
#     def init_weights(m):
#         if type(m) == nn.Linear or type(m) == nn.Conv2d:
#             nn.init.xavier_uniform_(m.weight)
#     autoencoder.apply(init_weights)
#     print('training on', device)
#     autoencoder.to(device)
#     optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr, weight_decay=1e-5)
#     loss_fn = nn.MSELoss()
#     timer, num_batches = ksd.Timer(), len(train_iter)
#     for epoch in range(num_epochs):
#         # 训练损失之和，训练准确率之和，样本数
#         print(f'Epoch {epoch + 1}\n--------------------------------')
#         metric = ksd.Accumulator(2)
#         autoencoder.train()
#         for i, (X, _) in enumerate(train_iter):
#             timer.start()
#             optimizer.zero_grad()
#             X = X.to(device)
#             X_noise = add_noise(X, noise_factor, device)
#             X_R = autoencoder(X_noise)
#             loss = loss_fn(X, X_R)
#             loss.backward()
#             optimizer.step()
#             with torch.no_grad():
#                 metric.add(loss * X.shape[0], X.shape[0])
#             timer.stop()
#             train_l = metric[0] / metric[1]
#             if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
#                 print(f'loss {train_l:.7f}')
#         test_loss = ksd.evaluate_loss_gpu(autoencoder, test_iter, loss_fn)
#         print(f'test loss {test_loss:.7f}')
#     print(f'{metric[1] * num_epochs / timer.sum():.1f} examples/sec '
#           f'on {str(device)}')
#     if model_name != "none":
#         torch.save(autoencoder.state_dict(), model_name)
#
#
# def show_images(decode_images, x_test):
#     """
#     plot the images.
#     :param decode_images: the images after decoding
#     :param x_test: testing data
#     :return:
#     """
#     n = 10
#     plt.figure(figsize=(20, 4))
#     for i in range(n):
#         ax = plt.subplot(2, n, i+1)
#         ax.imshow(x_test[i].reshape(28, 28))
#         plt.gray()
#         ax.get_xaxis().set_visible(False)
#         ax.get_yaxis().set_visible(False)
#
#         ax = plt.subplot(2, n, i + 1 + n)
#         ax.imshow(decode_images[i].detach().numpy().reshape(28, 28))
#         plt.gray()
#         ax.get_xaxis().set_visible(False)
#         ax.get_yaxis().set_visible(False)
#     plt.show()
#
#
#
#
#
#
#
# loss_fn = torch.nn.MSELoss()
# lr = 1e-3
# torch.manual_seed(0)
# d = 4
# epochs = 50
#
# net = ConvAutoEncoder(encoded_space_dim=10, fc2_input_dim=128)
#
# train_autoencoder_and_save(net, train_loader, test_loader, epochs, lr, noise_factor=0.1, device=ksd.try_gpu(), model_name='./models/autoencoders/conv_autoencoder_mnist.pth')
# for X, y in test_loader:
#     show_images(net(X.to("cuda")).to('cpu'), X)
#     break
import torch
print(torch.cuda.device_count())
