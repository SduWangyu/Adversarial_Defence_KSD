default_label_file = '../data/ImageNet/LOC_synset_mapping.txt'

# mean and std of cifar100 dataset
cifar10_train_mean = (0.4914, 0.4822, 0.4465)
cifar10_train_std = (0.2023, 0.1994, 0.2010)
cifar10_test_mean = [0.4914, 0.4822, 0.4465]
cifar10_test_std = [0.2023, 0.1994, 0.2010]

cifar100_train_mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
cifar100_train_std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
cifar100_test_mean = (0.5088964127604166, 0.48739301317401956, 0.44194221124387256)
cifar100_test_std = (0.2682515741720801, 0.2573637364478126, 0.2770957707973042)

dataset_root = '../dataset/'
# total training epoches

EPOCH = 200
MILESTONES = [60, 120, 160]

thread_num = 10

shift = 2
