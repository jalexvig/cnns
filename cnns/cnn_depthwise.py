import numpy as np

from cnns.utils import pad_image, convert_to_torch


def cnn2d_depthwise(image: np.ndarray,
                    filters: np.ndarray):
    """
    Depthwise convolutions.
    Args:
        image: (hi, wi, cin).
        filters: (hf, wf, cin, cmul).

    Returns:
        (hi, wi, cin * cmul)
    """

    filter_len, image_padded = pad_image(filters, image)

    n_cout_cin = filters.shape[-1]

    out = np.zeros([image.shape[0], image.shape[1], image.shape[2] * n_cout_cin])

    for idx_cin in range(image.shape[2]):
        for idx_row in range(image.shape[0]):
            for idx_col in range(image.shape[1]):
                patch = image_padded[idx_row: idx_row + filter_len, idx_col: idx_col + filter_len, idx_cin]
                for idx_cmul in range(n_cout_cin):
                    filter = filters[:, :, idx_cin, idx_cmul]
                    out[idx_row, idx_col, idx_cin * n_cout_cin + idx_cmul] = (filter * patch).sum()

    return out


def cnn2d_depthwise_torch(image: np.ndarray,
                          filters: np.ndarray):

    from torch.nn import functional as F

    image_torch, filters_torch = convert_to_torch(image, filters)

    df, _, cin, cmul = filters.shape
    filters_torch = filters_torch.transpose(0, 1).contiguous()
    filters_torch = filters_torch.view(cin * cmul, 1, df, df)

    features_torch = F.conv2d(image_torch, filters_torch, padding=df // 2, groups=cin)
    features_torch_ = features_torch.numpy()[0].transpose([2, 1, 0])

    return features_torch_


def cnn2d_depthwise_tf(image: np.ndarray,
                       filters: np.ndarray):

    import tensorflow as tf
    tf.enable_eager_execution()

    features_tf = tf.nn.depthwise_conv2d(image[None], filters, strides=[1, 1, 1, 1], padding='SAME')

    return features_tf


if __name__ == '__main__':

    from cnns.settings import H_IMG, W_IMG, CIN, FILTER_LEN, CMUL

    # Create inputs
    image = np.random.rand(H_IMG, W_IMG, CIN)
    filters = np.random.rand(FILTER_LEN, FILTER_LEN, CIN, CMUL)

    # Convolve
    features_np = cnn2d_depthwise(image, filters)

    # Compare to pytorch
    features_torch = cnn2d_depthwise_torch(image, filters)
    print('Pytorch:', np.isclose(features_np, features_torch).all())

    # Compare to tensorflow
    features_tf = cnn2d_depthwise_tf(image, filters)
    print('Tensorflow:', np.isclose(features_tf[0], features_np).all())
