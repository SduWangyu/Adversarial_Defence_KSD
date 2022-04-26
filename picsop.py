import torch
import numpy as np
from PIL import Image


class MapIdToHumanString:
    def __init__(self, label_filename=None):
        if not label_filename:
            label_filename = './data/ImageNet/LOC_synset_mapping.txt'
        self.label_lookup = self.load(label_filename)

    def load(self, label_filename):
        label_list = []
        with open(label_filename) as f:
            for line in f:
                label_list.append(line[10:-1:].split(',')[0])

        return label_list.copy()

    def get_label(self, node_id):
        if node_id >= len(self.label_lookup) or node_id < 0:
            return ''
        return self.label_lookup[node_id]


def transform_reverse(ori_image_tensor, totensor_transform):
    """
    reverse
    :param ori_image_tensor:
    :param totensor_transform:
    :return:
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image_tensor = ori_image_tensor.clone().detach().numpy().squeeze().transpose(1, 2, 0)
    if 'Normalize' in str(totensor_transform):
        image_tensor = (image_tensor * std) + mean
    if 'ToTensor' in str(totensor_transform) or image_tensor.max() < 1.0:
        image_tensor = image_tensor * 255.0

    return image_tensor


def transform_to_img_from_dataset(ori_image_tensor, totensor_transform, pic_name):
    """
    从image_tensor到img
    param img_tensor: tensor
    param transforms: torchvision.transforms
    param pic_name: pic path and name
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image_tensor = ori_image_tensor.clone().detach().numpy().squeeze().transpose(1, 2, 0)
    if 'Normalize' in str(totensor_transform):
        image_tensor = (image_tensor * std) + mean
    if 'ToTensor' in str(totensor_transform) or image_tensor.max() < 1.0:
        image_tensor = image_tensor * 255.0

    if image_tensor.shape[2] == 3:
        image = Image.fromarray(image_tensor.astype('uint8')).convert('RGB')
    elif image_tensor.shape[2] == 1:
        image_tensor = image_tensor[0]
        image = Image.fromarray(image_tensor.astype('uint8')).squeeze()
    else:
        raise Exception("Invalid img shape, expected 1 or 3 in axis 2, but got {}!".format(image_tensor.shape[2]))
    image.save(pic_name)
    return image


def image_preprocessing(image_path, image_trasforms, is_unsqueeze=False):
    """
    根据路径对图片进行预处理
    :param is_unsqueeze: 是否加一列
    :param image_path:图片的路径
    :param image_trasforms:torch.transforms
    :return:
    """

    img = Image.open(image_path)
    if is_unsqueeze:
        image_tensor = image_trasforms(img).unsqueeze(0)
    else:
        image_tensor = image_trasforms(img)
    return image_tensor

map_id_to_label_string = MapIdToHumanString()
def show_images_diff(original_img_tensor, adversarial_img_tensor, ToTensor_TransForms, original_label, adversial_label):
    """
    对比展现原始图片和对抗样本图片
    :param original_label:
    :param ToTensor_TransForms:
    :param original_img:
    :param adversarial_img:
    :return:
    """

    import matplotlib.pyplot as plt
    original_img = transform_reverse(original_img_tensor, ToTensor_TransForms)
    adversarial_img = transform_reverse(adversarial_img_tensor, ToTensor_TransForms)
    original_img = np.clip(original_img, 0, 255).astype(np.uint8)
    adversarial_img = np.clip(adversarial_img, 0, 255).astype(np.uint8)
    if original_img.any() > 1.0:
        original_img = original_img / 255.0
    if adversarial_img.any() > 1.0:
        adversarial_img = adversarial_img / 255.0
    plt.figure()

    plt.subplot(131)
    plt.title('Original')
    plt.title('lable:{}'.format(map_id_to_label_string.get_label(original_label)), y=-0.16, loc="right")
    plt.imshow(original_img)
    plt.axis('off')

    plt.subplot(132)
    plt.title('Adversarial')
    plt.imshow(adversarial_img)
    plt.title('lable:{}'.format(map_id_to_label_string.get_label(adversial_label)), y=-0.16, loc="right")
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

if __name__ == '__main__':
    map_id_to_label_string.get_label(288)

