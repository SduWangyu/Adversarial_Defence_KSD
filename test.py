import core.classifiers as clfs
from core.custom_dataset import load_data
from train import train_classifier
from torchvision import transforms
from utils.misc import try_gpu

if __name__ == "__main__":
    net = clfs.ResNet50CIFAR100()
    trans = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                                                     (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))])
    train_iter, test_iter = load_data("cifar100", batch_size=128, trans=trans)
    train_classifier(net, train_iter, test_iter, num_epochs=200, lr=0.01, device=try_gpu())
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
