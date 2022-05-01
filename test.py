import core.classifiers as clfs
from core.custom_dataset import load_data
from train import train_classifier
from torchvision import transforms
from utils.misc import try_gpu
import utils.constants as cst

if __name__ == "__main__":
    net = clfs.ResNet18CIFAR10()
    trans_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(cst.cifar10_train_mean,
                                                           cst.cifar10_train_std)])
    trans_test = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(cst.cifar10_train_mean,
                                                          cst.cifar10_train_std)])
    train_iter, test_iter = load_data("cifar10", batch_size=32, transforms_train=trans_train, transforms_test=trans_test)
    train_classifier(net, train_iter, test_iter, num_epochs=2, lr=0.01,
                     device=try_gpu(), model_name='./models/classifiers/resnet18_cifar10.pth')
    # trans = transforms.Compose([transforms.ToTensor()])
    # data_iter = load_data("mnist", batch_size=1, val_type=2, trans=trans, val_account=0.002)
    #
    # device = misc.try_gpu()
    # net = clfs.ClassiferMNIST()
    # net.load_exist()
    # net.to(device)
    # for X, y in data_iter:
    #     X, y = X.to(device), y.to(device)
    #     X_adv, y_adv = ae.cw_l2(net, X, y, device=device, k=0)
    #     break
