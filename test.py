import core.classifiers as clfs
import core.autoencoders as ae
from core.custom_dataset import load_data
from train import train_classifier, train_autoencoder
from torchvision import transforms
from utils.misc import try_gpu
import utils.constants as cst


if __name__ == "__main__":
    net = ae.ResAutoEncoderCIFAR100()
    trans_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(cst.cifar100_train_mean,
                                                           cst.cifar100_train_std)])
    trans_test = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(cst.cifar100_train_mean,
                                                          cst.cifar100_train_std)])
    train_iter, test_iter = load_data("cifar100", batch_size=16, transforms_train=trans_train, transforms_test=trans_test)
    # train_classifier(net, train_iter, test_iter, num_epochs=2, lr=0.01,
    #                  device=try_gpu(), model_name='./data/models_trained/classifiers/resnet18_cifar10.pth')
    train_autoencoder(net, train_iter, test_iter, num_epochs=50, lr=0.01,
                     device=try_gpu(), model_name='./data/models_trained/autoencoders/resnet50_cifar100.pth')
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
    #     brea