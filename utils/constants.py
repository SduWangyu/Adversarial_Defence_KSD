default_label_file = '../data/ImageNet/LOC_synset_mapping.txt'

# mean and std of cifar100 dataset
cifar10_train_mean = (0.4914, 0.4822, 0.4465)
cifar10_train_std = (0.2023, 0.1994, 0.2010)
cifar10_test_mean = (0.4914, 0.4822, 0.4465)
cifar10_test_std = (0.2023, 0.1994, 0.2010)

dataset_root = '../dataset/'
# total training epoches

EPOCH = 200
MILESTONES = [60, 120, 160]