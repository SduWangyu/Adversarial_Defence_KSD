import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from utils import constants


class MapIdToHumanString:
    label_list = []

    def __init__(self, label_filename=None):
        pass

    @classmethod
    def load_file(cls, label_filename):
        MapIdToHumanString.label_list = []
        with open(label_filename) as f:
            for line in f:
                MapIdToHumanString.label_list.append(line[10:-1:].split(',')[0])

    @classmethod
    def get_label(cls, node_id):
        if not MapIdToHumanString.label_list:
            MapIdToHumanString.label_list = None
            MapIdToHumanString.load_file(constants.default_label_file)
        if node_id >= len(MapIdToHumanString.label_list) or node_id < 0:
            return ''
        return MapIdToHumanString.label_list[node_id]


def transform_reverse(ori_image_tensor, to_tensor_transform, pic_name=None):
    """
    从image_tensor到img
    param img_tensor: tensor
    param transforms: torchvision.transforms
    param pic_name: pic path and name
    """
    mean = constants.cifar10_test_mean
    std = constants.cifar10_test_std
    image_tensor = ori_image_tensor.clone().detach().numpy().transpose(1, 2, 0)
    if 'Normalize' in str(to_tensor_transform):
        image_tensor = (image_tensor * std) + mean
    if 'ToTensor' in str(to_tensor_transform) or image_tensor.max() < 1.0:
        image_tensor = image_tensor * 255.0

    if pic_name:
        if image_tensor.shape[2] == 3:
            image = Image.fromarray(image_tensor.astype('uint8')).convert('RGB')
        elif image_tensor.shape[2] == 1:
            image_tensor = image_tensor[0]
            image = Image.fromarray(image_tensor.astype('uint8')).squeeze()
        else:
            raise Exception("Invalid img shape, expected 1 or 3 in axis 2, but got {}!".format(image_tensor.shape[2]))
        image.save(pic_name)

    return image_tensor


def image_preprocessing(image_path, image_transforms, is_unsqueeze=False):
    """
    根据路径对图片进行预处理
    :param is_unsqueeze: 是否加一列
    :param image_path:图片的路径
    :param image_transforms:torch.transforms
    :return:
    """

    img = Image.open(image_path)
    if is_unsqueeze:
        image_tensor = image_transforms(img).unsqueeze(0)
    else:
        image_tensor = image_transforms(img)
    return image_tensor


def show_images_diff(original_img_tensor, adversarial_img_tensor, to_tensor_transforms, original_label,
                     adversial_label):
    """
    对比展现原始图片和对抗样本图片
    :param original_img_tensor:
    :param adversarial_img_tensor:
    :param to_tensor_transforms:
    :param original_label:
    :param adversial_label:
    :return:
    """
    original_img = transform_reverse(original_img_tensor, to_tensor_transforms)
    adversarial_img = transform_reverse(adversarial_img_tensor, to_tensor_transforms)
    original_img = np.clip(original_img, 0, 255).astype(np.uint8)
    adversarial_img = np.clip(adversarial_img, 0, 255).astype(np.uint8)
    if original_img.any() > 1.0:
        original_img = original_img / 255.0
    if adversarial_img.any() > 1.0:
        adversarial_img = adversarial_img / 255.0
    plt.figure()

    plt.subplot(131)
    plt.title('Original')
    plt.title('label:{}'.format(MapIdToHumanString.get_label(original_label)), y=-0.16, loc="right")
    plt.imshow(original_img)
    plt.axis('off')

    plt.subplot(132)
    plt.title('Adversarial')
    plt.imshow(adversarial_img)
    plt.title('label:{}'.format(MapIdToHumanString.get_label(adversial_label)), y=-0.16, loc="right")
    plt.axis('off')

    plt.subplot(133)
    plt.title('Adversarial-Original')
    difference = adversarial_img - original_img
    # (-1,1)  -> (0,1)
    difference = difference / abs(difference).max() / 2.0 + 0.5
    plt.imshow(difference, cmap=plt.cm.gray)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def show_images_ae(original_images, noised_images, decoded_images):
    """
    plot the images.
    :param original_images: The original image
    :param noised_images:
    :param decoded_images: The images after decoding
    :return: None
    """
    n = 10
    mean = constants.cifar10_test_mean
    std = constants.cifar10_test_std
    plt.figure(figsize=(20, 4))
    for i in range(n):
        tmp_org = np.clip((original_images[i].detach().clone().numpy().transpose(1, 2, 0) * std) + mean, 0, 1)
        tmp_adv = np.clip((noised_images[i].detach().clone().numpy().transpose(1, 2, 0) * std) + mean, 0, 1)
        tmp_rec = decoded_images[i].detach().clone().numpy().transpose(1, 2, 0)
        ax = plt.subplot(5, n, i + 1)
        print(tmp_rec.shape, tmp_org.shape, tmp_adv.shape)
        ax.imshow(tmp_org)

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(5, n, i + 1 + n)
        ax.imshow(tmp_adv)

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        difference = tmp_org - tmp_adv
        # (-1,1)  -> (0,1)
        difference = difference / abs(difference).max() / 2.0 + 0.5

        ax = plt.subplot(5, n, i + 1 + 2 * n)
        ax.imshow(difference)

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        difference = tmp_org - tmp_rec
        # (-1,1)  -> (0,1)
        difference = difference / abs(difference).max() / 2.0 + 0.5

        ax = plt.subplot(5, n, i + 1 + 4 * n)
        ax.imshow(difference)

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(5, n, i + 1 + 3 * n)
        ax.imshow(tmp_rec)

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()


def show_images_test(original_images, noised_images, decoded_images, n=5):
    """
    plot the images.
    :param original_images: The original image
    :param noised_images:
    :param decoded_images: The images after decoding
    :return: None
    """

    mean = constants.cifar10_test_mean
    std = constants.cifar10_test_std
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # tmp_org = np.clip((original_images[i].detach().clone().numpy().transpose(1, 2, 0) * std) + mean, 0, 1)
        # tmp_adv = np.clip((noised_images[i].detach().clone().numpy().transpose(1, 2, 0) * std) + mean, 0, 1)
        # tmp_rec = np.clip((decoded_images[i].detach().clone().numpy().transpose(1, 2, 0) * std) + mean, 0, 1)
        tmp_org = original_images[i].detach().clone().numpy().transpose(1, 2, 0)
        tmp_adv = noised_images[i].detach().clone().numpy().transpose(1, 2, 0)
        tmp_rec = np.clip(decoded_images[i].detach().clone().numpy().transpose(1, 2, 0), 0, 1)
        ax = plt.subplot(5, n, i + 1)
        # print(tmp_rec.shape, tmp_org.shape, tmp_adv.shape)
        ax.imshow(tmp_org)

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(5, n, i + 1 + n)
        ax.imshow(tmp_adv)

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        difference = (tmp_org - tmp_adv) / 2.0 + 0.5
        # (-1,1)  -> (0,1)
        print(difference)
        difference = np.clip(difference, 0, 1)

        ax = plt.subplot(5, n, i + 1 + 2 * n)
        ax.imshow(difference)

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(5, n, i + 1 + 3 * n)
        ax.imshow(tmp_rec)

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        difference = (tmp_org - tmp_rec) / 2.0 + 0.5
        print(difference)
        # (-1,1)  -> (0,1)
        difference = np.clip(difference, 0, 1)

        ax = plt.subplot(5, n, i + 1 + 4 * n)
        ax.imshow(difference)

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)



    plt.show()
