import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
from torchvision import datasets


def transform_to_img_from_dataset(image_tensor, transform, pic_name):
    """
    param img_tensor: tensor
    param transforms: torchvision.transforms
    param pic_name: pic path and name
    """
    if 'Normalize' in str(transform):
        normal_transform = list(filter(lambda x: isinstance(x, transforms.Normalize), transform.transforms))
        mean = torch.tensor(normal_transform[0].mean, dtype=image_tensor.dtype, device=image_tensor.device)
        std = torch.tensor(normal_transform[0].std, dtype=image_tensor.dtype, device=image_tensor.device)
        image_tensor.mul_(std[:, None, None]).add_(mean[:, None, None])

    if 'ToTensor' in str(transform) or image_tensor.max() < 1:
        image_tensor = image_tensor.detach().numpy() * 255

    if isinstance(image_tensor, torch.Tensor):
        image_tensor = image_tensor.numpy().squeeze()

    if image_tensor.shape[0] == 3:
        image_tensor = image_tensor.transpose(1, 2, 0)
        image = Image.fromarray(image_tensor.astype('uint8')).convert('RGB')
    elif image_tensor.shape[0] == 1:
        image_tensor = image_tensor[0]
        image = Image.fromarray(image_tensor.astype('uint8'))
    else:
        raise Exception("Invalid img shape, expected 1 or 3 in axis 2, but got {}!".format(image_tensor.shape[2]))
    image.save(pic_name)
    return image


if __name__ == "__main__":
    train_data = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )

    # img = Image.open('./pics/test.png')
    img_tensor = train_data[8][0]
    print(img_tensor.shape)
    transform_to_img_from_dataset(img_tensor, ToTensor, './pics/test5.png')
