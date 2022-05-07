import torch
from torch.utils.data import DataLoader
from core.defence import decrease_color_depth, denoising_by_cv2, denoising_by_NlMeans
import core.attack_method as am
import core.autoencoders as ae
from core.custom_dataset import CustomDataset
import core.classifiers
from core.custom_dataset import load_data
from train import train_autoencoder
from torchvision import transforms
from utils.misc import try_gpu
from utils.image_operator import show_images_test, show_one_images


if __name__ == "__main__":
    # net = ae.ConvAutoEncoderCIFAR10()
    # trans_train = transforms.Compose([
    #                                   # transforms.RandomCrop(32, padding=4),
    #                                   # transforms.RandomHorizontalFlip(),
    #                                   transforms.ToTensor()])
    trans_test = transforms.Compose([transforms.ToTensor()
                                     ])
    # train_iter, test_iter = load_data("cifar10", batch_size=128, transforms_train=trans_train, transforms_test=trans_test)
    # # train_classifier(net, train_iter, test_iter, num_epochs=2, lr=0.01,
    # #                  device=try_gpu(), model_name='./data/models_trained/classifiers/resnet18_cifar10.pth')
    # train_autoencoder(net, train_iter, test_iter, num_epochs=100, lr=0.01,
    #                   device=try_gpu(), model_name='./data/models_trained/autoencoders/cov_cifar10_test.pth')
    # # net = core.classifiers.ResNet18CIFAR10()
    # net.load_exist(path="data/models_trained/classifiers/resnet18_cifar10.pth")
    # correct = 0
    # total_data = 0
    # for X, y in test_iter:
    #     res_y = net(X).argmax(1)
    #     for i in range(len(y)):
    #         correct += (y[i] == res_y[i])
    #         total_data += 1
    #         if total_data == 5000:
    #             break
    #         if not total_data % 100:
    #             print(total_data)
    #     else:
    #         continue
    #     break
    # print(total_data, correct, correct / total_data)
    device = try_gpu()
    cl = core.classifiers.ResNet18CIFAR10()
    cl.load_exist(path='./data/models_trained/classifiers/resnet18_cifar10.pth')
    cl.to(device)
    # net = ae.ConvLargeAutoEncoderCIFAR10()
    # net.to(device)
    # net.load_exist(path='data/models_trained/autoencoders/cov_cifar10_test_32.pth')
    net = denoising_by_cv2
    # dataset_adv = torch.load(f"./data/validation_data/validation_data_cifar10_cw_10_adv.pt")
    # data_adv_iter = DataLoader(dataset_adv, batch_size=10)
    # dataset_org = torch.load(f"./data/validation_data/validation_data_cifar10_cw_10_org.pt")
    # data_org_iter = DataLoader(dataset_org, batch_size=10)
    data_iter = load_data("cifar10", batch_size=1, val_type=2, transforms_train=trans_test, transforms_test=trans_test)
    tmp_adv = torch.rand(10, 3, 32, 32).to(device)
    tmp_X = torch.rand(10, 3, 32, 32).to(device)
    tmp_count = 0
    n = 6
    e = 0.03
    chk = True
    for X, _ in data_iter:
        chk = True
        X = X.to(device)

        y = _.to(device)
        # X_rec = net(X).cpu()

        # for i in range(10):
        while True:
            y_pre_org = cl(X).argmax(1)

            if y_pre_org != y:
                chk = False
                break

            tmp = am.fgsm_i(cl, X, y, eps=e, device=device)
            tmp_adv[tmp_count], adv_label = tmp[0].detach().clone().squeeze(0).to(device), tmp[1]
            if adv_label != y:
                # show_one_images((X.cpu(), tmp_adv[tmp_count].detach().clone().unsqueeze(0).cpu()), in_net=net, e_list=[i for i in range(8, 0, -1)])
                # exit()
                # print(y_pre_org, y, adv_label)
                break
        if chk:
            tmp_X[tmp_count] = X.detach().clone().squeeze(0)
            tmp_count += 1
            e += 0.01
        if tmp_count == n:
            break
    X_adv_rec = net(tmp_adv).cpu()
    # print(tmp_adv[0])
    show_images_test(tmp_X.cpu(), tmp_adv.cpu(), X_adv_rec.cpu(), n=n)

