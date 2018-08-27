import numpy as np
import torch


def pad_image(filters, image):

    assert filters.shape[0] == filters.shape[1]
    assert filters.shape[0] % 2

    filter_len = filters.shape[0]
    pad = filter_len // 2

    image_padded = np.pad(image, [(pad,), (pad,), (0,)], 'constant')

    return filter_len, image_padded


def convert_to_torch(image, filters):

    image_torch = torch.tensor(image.transpose([2, 1, 0])[None])
    filters_torch = torch.tensor(filters.transpose([3, 2, 1, 0]))

    return image_torch, filters_torch
