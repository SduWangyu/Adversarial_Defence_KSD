

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
