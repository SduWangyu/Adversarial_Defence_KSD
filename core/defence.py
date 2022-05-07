import torch
import core.autoencoders as autoencoders
from torchvision.transforms import ToTensor, Compose
import cv2

import train
import utils.misc
import utils.misc as misc
from core.custom_dataset import *
from core.autoencoders import *
from core.deployed_defensive_model import DefenceCIFAR10, DefenceMNIST
from utils import evaluate



def get_thread_hold_by_distance(dataset_name, drop_rate=0.01, p=1, device=None):
    if dataset_name == "mnist":
        autoencoder = ConvAutoEncoderMNIST()
        autoencoder.load_exist(path="../data/models_trained/autoencoders/cov_mnist.pth")
        autoencoder.to(device)
    if dataset_name == "cifar10":
        autoencoder = ConvAutoEncoderCIFAR10()
        autoencoder.load_exist(path="../data/models_trained/autoencoders/cov_cifar10_colab_noise_factor_001.pth")
        autoencoder.to(device)
    validation_data_iter = load_data(dataset_name, val_type=2, val_account=0.1, batch_size=1,
                                     transforms_train=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]),
                                     transforms_test=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))
    lp_distance = []
    autoencoder = decrease_color_depth
    for X, _ in validation_data_iter:
        X = X.to(device)
        X_r = autoencoder(X)
        lp_distance.append(torch.dist(X, X_r, p=p).cpu().detach().clone())
    print(len(lp_distance))
    lp_distance = torch.Tensor(lp_distance)
    thr, indices = torch.topk(lp_distance, int(drop_rate * lp_distance.shape[0]), dim=0)
    print(thr)
    return



def get_thread_hold_squeeze_lp(dataset_name, attack_method, drop_rate=0.01, device=None):
    autoencoder = []

    autoencoder.append(denoising_by_cv2)
    if dataset_name == "mnist":
        tmp_ae = ConvAutoEncoderMNIST()
        tmp_ae.load_exist(path="../data/models_trained/autoencoders/cov_mnist.pth")
        tmp_ae.to(device)
        autoencoder.append(tmp_ae)
        classifier = clfs.ClassiferMNIST()
        classifier.load_exist(path="../data/models_trained/classifiers/classifier_mnist.pth")
        classifier.to(device)
    elif dataset_name == "cifar10":
        tmp_ae = ConvLargeAutoEncoderCIFAR10()
        tmp_ae.load_exist(path="../data/models_trained/autoencoders/cov_cifar10_test_32.pth")
        tmp_ae.to(device)
        autoencoder.append(tmp_ae)
        classifier = clfs.ResNet18CIFAR10()
        classifier.load_exist(path="../data/models_trained/classifiers/resnet18_cifar10.pth")
        classifier.to(device)
    # i_i = [1, 5, 10]
    i_i = range(0, 50, 10)
    autoencoder.append(decrease_color_depth)
    autoencoder = [autoencoder[2]]
    for i in i_i:
        print(i)
        print()
        dataset_org = torch.load(f"../data/validation_data/validation_data_{dataset_name}_{attack_method}_{i}_org.pt")
        data_iter_org = DataLoader(dataset_org, batch_size=200)
        l2_distance = []
        for X, _ in data_iter_org:
            X = X.type(torch.float32).to(device)
            X_rec = X
            for ae in autoencoder:
                X_rec = ae(X_rec)

            for x, x_rec in zip(X, X_rec):
                l2_distance.append(evaluate.jenson_shannon_divergence(classifier(x.unsqueeze(0)), classifier(x_rec.unsqueeze(0))).cpu().detach().clone())
                # l2_distance.append(torch.dist(x, x_rec, p=p).cpu().detach().clone())
            break
        l2_distance_tensor = torch.Tensor(l2_distance)
        thr, indices = torch.topk(l2_distance_tensor, 200, dim=0)
        print(thr)

        dataset_adv = torch.load(f"../data/validation_data/validation_data_{dataset_name}_{attack_method}_{i}_adv.pt")
        data_iter_adv = DataLoader(dataset_adv, batch_size=200)
        # data_iter_org = load_data("cifar10", batch_size=15)
        l2_distance = []
        for X, _ in data_iter_adv:
            X = X.type(torch.float32).to(device)
            X_rec = X
            for ae in autoencoder:
                X_rec = ae(X_rec)

            for x, x_rec in zip(X, X_rec):
                l2_distance.append(evaluate.jenson_shannon_divergence(classifier(x.unsqueeze(0)), classifier(x_rec.unsqueeze(0))).cpu().detach().clone())
                # l2_distance.append(torch.dist(x, x_rec, p=p).cpu().detach().clone())
            break
        l2_distance_tensor = torch.Tensor(l2_distance)
        thr, indices = torch.topk(l2_distance_tensor, 200, dim=0)
        print(thr)

        # dataset_org = torch.load("./data/validation_data/validation_data_mnist_fgsm_org_eps_0{}.pt".format(i))
        # data_iter_org = data.DataLoader(dataset_adv, batch_size=200)


def decrease_color_depth(org_image):
    device = org_image.data.device
    res_image = torch.zeros(org_image.shape).to(device)
    for i in range(res_image.shape[0]):
        tmp_image = misc.copy_tensor(org_image[i].cpu()).numpy().transpose(1, 2, 0)
        tmp_image = ((tmp_image * 255) // (2 ** (8 - constants.shift))) / (2 ** constants.shift)
        res_image[i] = torch.Tensor(tmp_image.transpose(2, 0, 1)).to(device)
    return res_image


def denoising_by_NlMeans(org_image, twindow_size, swindow_size):
    device = org_image.data.device
    res_image = torch.zeros(org_image.shape).to(device)
    for i in range(res_image.shape[0]):
        tmp_image = org_image[i].cpu().detach().clone().numpy().transpose(1, 2, 0)
        tmp_image = cv2.cvtColor(tmp_image, cv2.COLOR_GRAY2BGR)
        tmp_image = cv2.fastNlMeansDenoisingColored(tmp_image, hColor=1, templateWindowSize=twindow_size, searchWindowSize=swindow_size)
        tmp_image = np.clip(tmp_image, 0, 1)
        # tmp_image = np.expand_dims(tmp_image, 2)
        res_image[i] = torch.Tensor(tmp_image.transpose(2, 0, 1)).to(device)
    return res_image


def denoising_by_median(org_image):
    device = org_image.data.device
    # print(org_image.shape)
    res_image = torch.zeros(org_image.shape).to(device)
    for i in range(org_image.shape[0]):
        # print(org_image[i].shape)
        tmp_image = misc.copy_tensor(org_image[i]).cpu().detach().clone().numpy().transpose(1, 2, 0)
        # print(tmp_image.shape)
        tmp_image = cv2.medianBlur(tmp_image, 3)
        # print(tmp_image.shape)
        tmp_image = np.clip(tmp_image, 0, 1)
        # tmp_image = np.expand_dims(tmp_image, 2)
        # print(tmp_image.shape)
        res_image[i] = torch.Tensor(tmp_image.transpose(2, 0, 1)).to(device)
    return res_image


def denoising_by_Gaussian(org_image):
    device = org_image.data.device
    # print(org_image.shape)
    res_image = torch.zeros(org_image.shape).to(device)
    for i in range(org_image.shape[0]):
        # print(org_image[i].shape)
        tmp_image = misc.copy_tensor(org_image[i]).cpu().detach().clone().numpy().transpose(1, 2, 0)
        # print(tmp_image.shape)
        tmp_image = cv2.GaussianBlur(tmp_image, (3, 3), 0)
        # print(tmp_image.shape)
        tmp_image = np.clip(tmp_image, 0, 1)
        # tmp_image = np.expand_dims(tmp_image, 2)
        # print(tmp_image.shape)
        res_image[i] = torch.Tensor(tmp_image.transpose(2, 0, 1)).to(device)
    return res_image


def reconstruct(org_tensor, net, noise_factor=0.1):
    tmp_tensor = org_tensor
    tmp_tensor = train.add_noise(tmp_tensor, noise_factor)
    tmp_tensor = net(tmp_tensor)
    res_tensor = tmp_tensor
    return res_tensor


def get_defence_without_clf_test_result(dataset_name, attack_method, drop_rate=0.1, noise_factor=0.1, device=None, test_batch_size=200):
    global fileName
    fileName = f'../res_data/res_no_clf_{dataset_name}-{attack_method}_drop_rate={drop_rate}.txt'

    print_file(f'{dataset_name}_{attack_method}: drop_rate={drop_rate}, noise_factor={noise_factor}')
    batch_size = test_batch_size

    reconstruct_method = [denoising_by_Gaussian]
    reconstructor = None

    if attack_method == 'fgsm_i':
        i_i = [1, 5, 10]
    elif attack_method == 'cw':
        i_i = range(0, 50, 10)
    elif attack_method == 'df':
        i_i = [1]
    else:
        raise Exception('No such dataset.')

    if dataset_name == "mnist":
        if attack_method == 'fgsm_i':
            i_i = range(1, 6)
        tmp_ae = ConvAutoEncoderMNIST()
        tmp_ae.load_exist(path="../data/models_trained/autoencoders/cov_mnist.pth")
        tmp_ae.to(device)
        reconstruct_method.append(tmp_ae)
        reconstructor = tmp_ae
        classifier = clfs.ClassiferMNIST()
        classifier.load_exist(path="../data/models_trained/classifiers/classifier_mnist.pth")
        classifier.to(device)
    elif dataset_name == "cifar10":
        tmp_ae = ConvLargeAutoEncoderCIFAR10()
        tmp_ae.load_exist(path="../data/models_trained/autoencoders/cov_cifar10_test_32.pth")
        tmp_ae.to(device)
        reconstruct_method.append(tmp_ae)

        tmp_ae = ConvLargeAutoEncoderCIFAR10New()
        tmp_ae.load_exist(path="../data/models_trained/autoencoders/cov_cifar10_test_32_new.pth")
        tmp_ae.to(device)
        reconstructor = tmp_ae

        classifier = clfs.ResNet18CIFAR10()
        classifier.load_exist(path="../data/models_trained/classifiers/resnet18_cifar10_test.pth")
        classifier.to(device)

    print_file(str([i.__name__ for i in reconstruct_method]))

    adv_distances = []
    adv_tensor = []
    adv_label = []
    org_distances = []
    org_tensor = None
    org_label = None

    data_iter = load_data(dataset_name, batch_size=batch_size, val_type=2, val_account=0.014,
                          transforms_test=Compose([ToTensor()]), transforms_train=Compose([ToTensor()]))

    for X, _ in data_iter:
        org_tensor = X.to(device)
        org_label = _.to(device)
        break

    for method in reconstruct_method:
        distance = []
        X = utils.misc.copy_tensor(org_tensor)

        X = X.type(torch.float32).to(device)
        X_rec = reconstruct(X, method, noise_factor)
        for x, x_rec in zip(X, X_rec):
            distance.append(torch.dist(x.unsqueeze(0), x_rec.unsqueeze(0)).cpu().detach().clone())
            # l2_distance.append(torch.dist(x, x_rec, p=p).cpu().detach().clone())
        distance_tensor = torch.tensor(distance)

        # thr_1, indices = torch.topk(distance_tensor, 200, dim=0)
        org_distances.append(distance_tensor)

    dataset_distances = []

    tmp_dataset = []
    for X, _ in data_iter:
        tmp_dataset.append(X.to(device))

    for method in reconstruct_method:
        distance = []

        for X in tmp_dataset:
            X = X.type(torch.float32).to(device)
            X_rec = reconstruct(X, method, noise_factor)
            for x, x_rec in zip(X, X_rec):
                distance.append(torch.dist(x.unsqueeze(0), x_rec.unsqueeze(0)).cpu().detach().clone())
                # l2_distance.append(torch.dist(x, x_rec, p=p).cpu().detach().clone())
        distance_tensor = torch.tensor(distance)

        # thr_1, indices = torch.topk(distance_tensor, 200, dim=0)
        dataset_distances.append(distance_tensor)

    get_T = torch.topk(torch.Tensor([i.cpu().detach().clone().numpy() for i in dataset_distances]).max(0).values, int(len(dataset_distances[0]) * drop_rate), dim=0)[0][-1]

    for i in i_i:

        tmp_adv_thr = []
        dataset_adv = torch.load(f"../data/validation_data/validation_data_{dataset_name}_{attack_method}_{i}_adv.pt")
        save_to_adv_tensor = True
        for method in reconstruct_method:

            data_iter_adv = DataLoader(dataset_adv, batch_size=batch_size)
            distance = []
            X = None
            y = None
            for X, y in data_iter_adv:
                break
            if X is None:
                raise Exception("data_iter_adv is empty.")
            if save_to_adv_tensor:
                adv_tensor.append(X.to(device))
                adv_label.append(y.to(device))
                save_to_adv_tensor = False
            X = X.type(torch.float32).to(device)
            X_rec = method(X)

            for x, x_rec in zip(X, X_rec):
                distance.append(torch.dist(x.unsqueeze(0), x_rec.unsqueeze(0)).cpu().detach().clone())
                # l2_distance.append(torch.dist(x, x_rec, p=p).cpu().detach().clone())
            distance_tensor = torch.tensor(distance)
            tmp_adv_thr.append(distance_tensor)
        adv_distances.append(tmp_adv_thr)
        # org_distances.append(tmp_org_thr)
        # dataset_org = torch.load("./data/validation_data/validation_data_mnist_fgsm_org_eps_0{}.pt".format(i))
        # data_iter_org = data.DataLoader(dataset_adv, batch_size=200)

    print('Got checking distances.')

    e = 0
    for each_e_adv_distance in adv_distances:

        tmp_adv_tensor = torch.Tensor([i.cpu().detach().clone().numpy() for i in each_e_adv_distance]).max(0).values

        adv_not_checked_out = adv_tensor[e][tmp_adv_tensor <= get_T]
        adv_not_checked_out_label = adv_label[e][tmp_adv_tensor <= get_T]
        print_file(f'{(i_i[e]):>02}\tAdv_acc :{(1 - len(adv_not_checked_out) / len(tmp_adv_tensor.data)) * 100 :.2f}%\t'
              f' - {len(tmp_adv_tensor.data) - len(adv_not_checked_out)} \t/ {len(tmp_adv_tensor.data)}')
        if len(adv_not_checked_out) == 0:
            print_file(f'{(i_i[e]):>02}\tAdv_fix : NaN\t - 0 \t/ 0 \tchecked out after reconstructed.')
        else:
            adv_reconstructed = reconstruct(adv_not_checked_out.type(torch.float), reconstructor, noise_factor)

            adv_rec_label = classifier(adv_reconstructed).argmax(1)
            adv_num_fixed = torch.count_nonzero(adv_rec_label == adv_not_checked_out_label)

            print_file(
                f'{(i_i[e]):>02}\tAdv_fix :{(adv_num_fixed / adv_reconstructed.shape[0]) * 100 :.2f}%\t'
                f' - {adv_num_fixed} \t/ {adv_reconstructed.shape[0]} \tchecked out after reconstructed.')

        e += 1
    tmp_org_tensor = torch.Tensor([i.cpu().detach().clone().numpy() for i in org_distances]).max(0).values
    org_mis_checked_out = org_tensor[tmp_org_tensor > get_T]
    org_mis_checked_out_label = org_label[tmp_org_tensor > get_T]
    print_file(
        f'\tOrg_acc :{(1 - len(org_mis_checked_out) / org_tensor.shape[0]) * 100 :.2f}%\t'
        f' - {org_tensor.shape[0] - len(org_mis_checked_out)} \t/ {org_tensor.shape[0]}')

    if len(org_mis_checked_out) == 0:
        print_file(
            f'\tOrg_fix : NaN\t - {0} \t/ {0} \tfixed after reconstructed.')
    else:
        org_reconstructed = reconstruct(org_mis_checked_out, reconstructor, noise_factor)
        # org_rec_distances = []
        # for org_mis_X, org_rec_X in zip(org_mis_checked_out, org_reconstructed):
        #     org_rec_distances.append(evaluate.jenson_shannon_divergence(classifier(org_mis_X.unsqueeze(0)), classifier(org_rec_X.unsqueeze(0))).cpu().detach().clone())
        #
        # org_recovered = org_mis_checked_out[torch.Tensor([i.cpu().detach().clone().numpy()[0] for i in org_rec_distances]) > get_T]
        org_rec_label = classifier(org_reconstructed).argmax(1)
        org_num_fixed = torch.count_nonzero(org_rec_label == org_mis_checked_out_label)
        # print(f'[{bin(select_ae)[2:]:>03}]\t\tOrg_fix :{(len(org_recovered) / len(org_mis_checked_out.data)) * 100 :.2f}%\t'
        #       f' - {len(org_recovered)} \t/ {len(org_mis_checked_out.data)} \tfixed after reconstructed.')
        print_file(
            f'\tOrg_fix :{(org_num_fixed / org_reconstructed.shape[0]) * 100 :.2f}%\t'
            f' - {org_num_fixed} \t/ {org_reconstructed.shape[0]} \tfixed after reconstructed.')

    tmp_org_tensor = torch.Tensor([i.cpu().detach().clone().numpy() for i in org_distances]).max(0).values
    org_checked_out = org_tensor[tmp_org_tensor <= get_T]
    org_checked_out_label = org_label[tmp_org_tensor <= get_T]
    if len(org_checked_out) == 0:
        print_file(
            f'\tOrg_err : NaN\t - {0} \t/ {0} \terror after reconstructed.')
    else:
        org_reconstructed = reconstruct(org_checked_out, reconstructor, noise_factor)

        org_rec_label = classifier(org_reconstructed).argmax(1)
        org_num_fixed = torch.count_nonzero(org_rec_label != org_checked_out_label)
        # print(f'[{bin(select_ae)[2:]:>03}]\t\tOrg_fix :{(len(org_recovered) / len(org_mis_checked_out.data)) * 100 :.2f}%\t'
        #       f' - {len(org_recovered)} \t/ {len(org_mis_checked_out.data)} \tfixed after reconstructed.')
        print_file(
            f'\tOrg_err :{(org_num_fixed / org_reconstructed.shape[0]) * 100 :.2f}%\t'
            f' - {org_num_fixed} \t/ {org_reconstructed.shape[0]} \terror after reconstructed.')
    pass


def select_with_3bits(select_bits: int, org_list):
    res_list = []
    if select_bits & 0b001:
        res_list.append(org_list[2])
    if select_bits & 0b010:
        res_list.append(org_list[1])
    if select_bits & 0b100:
        res_list.append(org_list[0])
    return res_list


def get_defence_test_result(dataset_name, attack_method, drop_rate=0.1, noise_factor=0.1, device=None, test_batch_size=200):

    global fileName
    fileName = f'../res_data/res_{dataset_name}-{attack_method}_drop_rate={drop_rate}.txt'

    print_file(f'{dataset_name}_{attack_method}: drop_rate={drop_rate}, noise_factor={noise_factor}')
    batch_size = test_batch_size

    reconstruct_method = None
    autoencoder = []
    autoencoder.append(denoising_by_Gaussian)
    autoencoder.append(decrease_color_depth)

    if attack_method == 'fgsm_i':
        i_i = [1, 5, 10]
    elif attack_method == 'cw':
        i_i = range(0, 50, 10)
    elif attack_method == 'df':
        i_i = [1]
    else:
        raise Exception('No such dataset.')

    if dataset_name == "mnist":
        if attack_method == 'fgsm_i':
            i_i = range(1, 6)
        tmp_ae = ConvAutoEncoderMNIST()
        tmp_ae.load_exist(path="../data/models_trained/autoencoders/cov_mnist.pth")
        tmp_ae.to(device)
        autoencoder.append(tmp_ae)
        reconstruct_method = tmp_ae
        classifier = clfs.ClassiferMNIST()
        classifier.load_exist(path="../data/models_trained/classifiers/classifier_mnist.pth")
        classifier.to(device)
    elif dataset_name == "cifar10":
        tmp_ae = ConvLargeAutoEncoderCIFAR10()
        tmp_ae.load_exist(path="../data/models_trained/autoencoders/cov_cifar10_test_32.pth")
        tmp_ae.to(device)
        autoencoder.append(tmp_ae)

        tmp_ae = ConvLargeAutoEncoderCIFAR10New()
        tmp_ae.load_exist(path="../data/models_trained/autoencoders/cov_cifar10_test_32_new.pth")
        tmp_ae.to(device)
        reconstruct_method = tmp_ae

        classifier = clfs.ResNet18CIFAR10()
        classifier.load_exist(path="../data/models_trained/classifiers/resnet18_cifar10_test.pth")
        classifier.to(device)

    print_file(str([i.__name__ for i in autoencoder]))

    adv_distances = []
    adv_tensor = []
    adv_label = []
    org_distances = []
    org_tensor = None
    org_label = None

    data_iter = load_data(dataset_name, batch_size=batch_size, val_type=2, val_account=0.014,
                         transforms_test=Compose([ToTensor()]), transforms_train=Compose([ToTensor()]))

    for X, _ in data_iter:
        org_tensor = X.to(device)
        org_label = _.to(device)
        break
    for ae in autoencoder:
        distance = []
        X = utils.misc.copy_tensor(org_tensor)

        X = X.type(torch.float32).to(device)
        X_ae = ae(X)

        for x, x_ae in zip(X, X_ae):
            distance.append(evaluate.jenson_shannon_divergence(classifier(x.unsqueeze(0)), classifier(
                x_ae.unsqueeze(0))).cpu().detach().clone())
            # l2_distance.append(torch.dist(x, x_rec, p=p).cpu().detach().clone())
        distance_tensor = torch.tensor(distance)

        # thr_1, indices = torch.topk(distance_tensor, 200, dim=0)
        org_distances.append(distance_tensor)

    dataset_distances = []

    tmp_dataset = []
    for X, _ in data_iter:
        tmp_dataset.append(X.to(device))

    for ae in autoencoder:
        distance = []

        for X in tmp_dataset:
            X = X.type(torch.float32).to(device)
            X_ae = ae(X)

            for x, x_ae in zip(X, X_ae):
                distance.append(evaluate.jenson_shannon_divergence(classifier(x.unsqueeze(0)), classifier(
                    x_ae.unsqueeze(0))).cpu().detach().clone())
                # l2_distance.append(torch.dist(x, x_rec, p=p).cpu().detach().clone())
        distance_tensor = torch.tensor(distance)

        # thr_1, indices = torch.topk(distance_tensor, 200, dim=0)
        dataset_distances.append(distance_tensor)

    for i in i_i:

        tmp_adv_thr = []
        dataset_adv = torch.load(f"../data/validation_data/validation_data_{dataset_name}_{attack_method}_{i}_adv.pt")
        save_to_adv_tensor = True
        for ae in autoencoder:

            data_iter_adv = DataLoader(dataset_adv, batch_size=batch_size)
            distance = []
            X = None
            y = None
            for X, y in data_iter_adv:
                break
            if X is None:
                raise Exception("data_iter_adv is empty.")
            if save_to_adv_tensor:
                adv_tensor.append(X.to(device))
                adv_label.append(y.to(device))
                save_to_adv_tensor = False
            X = X.type(torch.float32).to(device)
            X_ae = ae(X)

            for x, x_ae in zip(X, X_ae):
                distance.append(evaluate.jenson_shannon_divergence(classifier(x.unsqueeze(0)), classifier(x_ae.unsqueeze(0))).cpu().detach().clone())
                # l2_distance.append(torch.dist(x, x_rec, p=p).cpu().detach().clone())
            distance_tensor = torch.tensor(distance)
            # thr_2, indices = torch.topk(distance_tensor, 200, dim=0)
            tmp_adv_thr.append(distance_tensor)
        adv_distances.append(tmp_adv_thr)
        # org_distances.append(tmp_org_thr)
        # dataset_org = torch.load("./data/validation_data/validation_data_mnist_fgsm_org_eps_0{}.pt".format(i))
        # data_iter_org = data.DataLoader(dataset_adv, batch_size=200)

    print('Got checking distances.')

    # select_list = range(1, 8)
    select_list = [1, 2, 4, 3, 5, 6, 7]

    for select_ae in select_list:
        e = 0
        tmp_T = []

        tmp_org_distances = select_with_3bits(select_ae, org_distances)
        tmp_data_distances = select_with_3bits(select_ae, dataset_distances)

        if len(tmp_org_distances) > 1:
            tmp_data_tensor = \
            torch.topk(torch.Tensor([i.cpu().detach().clone().numpy() for i in tmp_data_distances]).min(0).values, int(len(dataset_distances[0]) * drop_rate), dim=0)[0]
            tmp_org_tensor = torch.Tensor([i.cpu().detach().clone().numpy() for i in tmp_org_distances]).min(0).values
            get_T = tmp_data_tensor[-1]

        else:
            tmp_data_tensor = torch.topk(torch.Tensor([i.cpu().detach().clone().numpy() for i in tmp_data_distances]).squeeze(0), int(len(dataset_distances[0]) * drop_rate), dim=0)
            tmp_org_tensor = torch.Tensor([i.cpu().detach().clone().numpy() for i in tmp_org_distances]).squeeze(0)
            get_T = tmp_data_tensor[0][-1]

        for each_cw_adv_distance in adv_distances:
            tmp_adv_distances = select_with_3bits(select_ae, each_cw_adv_distance)
            if len(tmp_adv_distances) > 1:
                tmp_adv_tensor = torch.Tensor([i.cpu().detach().clone().numpy() for i in tmp_adv_distances]).min(0).values
            else:
                tmp_adv_tensor = torch.Tensor([i.cpu().detach().clone().numpy() for i in tmp_adv_distances]).squeeze(0)

            tmp_T.append(get_T)

            adv_not_checked_out = adv_tensor[e][tmp_adv_tensor <= get_T]
            adv_not_checked_out_label = adv_label[e][tmp_adv_tensor <= get_T]
            print_file(f'[{bin(select_ae)[2:]:>03}]-{(i_i[e]):>02}\tAdv_acc :{(1 - len(adv_not_checked_out) / len(tmp_adv_tensor.data)) * 100 :.2f}%\t'
                  f' - {len(tmp_adv_tensor.data) - len(adv_not_checked_out)} \t/ {len(tmp_adv_tensor.data)}')
            if len(adv_not_checked_out) == 0:
                print_file(f'[{bin(select_ae)[2:]:>03}]-{(i_i[e]):>02}\tAdv_fix : NaN\t - 0 \t/ 0 \tchecked out after reconstructed.')
            else:
                adv_reconstructed = reconstruct(adv_not_checked_out.type(torch.float), reconstruct_method, noise_factor)

                adv_rec_label = classifier(adv_reconstructed).argmax(1)
                adv_num_fixed = torch.count_nonzero(adv_rec_label == adv_not_checked_out_label)

                print_file(
                    f'[{bin(select_ae)[2:]:>03}]-{(i_i[e]):>02}\tAdv_fix :{(adv_num_fixed / adv_reconstructed.shape[0]) * 100 :.2f}%\t'
                    f' - {adv_num_fixed} \t/ {adv_reconstructed.shape[0]} \tchecked out after reconstructed.')

            e += 1

        org_mis_checked_out = org_tensor[tmp_org_tensor > get_T]
        org_mis_checked_out_label = org_label[tmp_org_tensor > get_T]
        print_file(f'[{bin(select_ae)[2:]:>03}]\t\tOrg_acc :{(1 - len(org_mis_checked_out) / len(tmp_org_tensor.data)) * 100 :.2f}%\t'
              f' - {len(tmp_org_tensor.data) - len(org_mis_checked_out)} \t/ {len(tmp_org_tensor.data)}')

        if len(org_mis_checked_out) == 0:
            print_file(
                f'[{bin(select_ae)[2:]:>03}]\t\tOrg_fix : NaN\t - {0} \t/ {0} \tfixed after reconstructed.')
        else:
            org_reconstructed = reconstruct(org_mis_checked_out, reconstruct_method, noise_factor)
            # org_rec_distances = []
            # for org_mis_X, org_rec_X in zip(org_mis_checked_out, org_reconstructed):
            #     org_rec_distances.append(evaluate.jenson_shannon_divergence(classifier(org_mis_X.unsqueeze(0)), classifier(org_rec_X.unsqueeze(0))).cpu().detach().clone())
            #
            # org_recovered = org_mis_checked_out[torch.Tensor([i.cpu().detach().clone().numpy()[0] for i in org_rec_distances]) > get_T]
            org_rec_label = classifier(org_reconstructed).argmax(1)
            org_num_fixed = torch.count_nonzero(org_rec_label == org_mis_checked_out_label)
            # print(f'[{bin(select_ae)[2:]:>03}]\t\tOrg_fix :{(len(org_recovered) / len(org_mis_checked_out.data)) * 100 :.2f}%\t'
            #       f' - {len(org_recovered)} \t/ {len(org_mis_checked_out.data)} \tfixed after reconstructed.')
            print_file(
                f'[{bin(select_ae)[2:]:>03}]\t\tOrg_fix :{(org_num_fixed / org_reconstructed.shape[0]) * 100 :.2f}%\t'
                f' - {org_num_fixed} \t/ {org_reconstructed.shape[0]} \tfixed after reconstructed.')


        org_checked_out = org_tensor[tmp_org_tensor <= get_T]
        org_checked_out_label = org_label[tmp_org_tensor <= get_T]
        if len(org_checked_out) == 0:
            print_file(
                f'[{bin(select_ae)[2:]:>03}]\t\tOrg_err : NaN\t - {0} \t/ {0} \terror after reconstructed.')
        else:
            org_reconstructed = reconstruct(org_checked_out, reconstruct_method, noise_factor)

            org_rec_label = classifier(org_reconstructed).argmax(1)
            org_num_fixed = torch.count_nonzero(org_rec_label != org_checked_out_label)
            # print(f'[{bin(select_ae)[2:]:>03}]\t\tOrg_fix :{(len(org_recovered) / len(org_mis_checked_out.data)) * 100 :.2f}%\t'
            #       f' - {len(org_recovered)} \t/ {len(org_mis_checked_out.data)} \tfixed after reconstructed.')
            print_file(
                f'[{bin(select_ae)[2:]:>03}]\t\tOrg_err :{(org_num_fixed / org_reconstructed.shape[0]) * 100 :.2f}%\t'
                f' - {org_num_fixed} \t/ {org_reconstructed.shape[0]} \terror after reconstructed.')

        print_file('-' * 20)


fileName = ''


def print_file(msg):
    global fileName
    print(msg)
    with open(fileName, 'a+') as f:
        f.write(msg + '\n')


if __name__ == "__main__":
    pass
    # print(f'shift: {constants.shift}')

    # get_thread_hold_by_distance("cifar10", device=misc.try_gpu())
    # test_defence_by_distance("cifar10", "cw", p=1, device=misc.try_gpu())
    # get_thread_hold_by_prodiv("mnist", "cw", device=misc.try_gpu())
    # data_iter = load_data("cifar10", batch_size=1, val_type=2, transforms_train=Compose([ToTensor()]), transforms_test=Compose([ToTensor()]))
    # for X, _ in data_iter:
    #     denoising_by_NlMeans(X, 3)
    #     break
    # for drop_rate in [0.1, 0.05, 0.01]:
    #     get_defence_without_clf_test_result("cifar10", "cw", drop_rate=drop_rate, device=misc.try_gpu(), test_batch_size=200, noise_factor=0.1)
    # get_thread_hold_squeeze_lp("cifar10", "cw", device=misc.try_gpu())
    # test_denfence_dataset("cifar10", "cw", device=misc.try_gpu())

