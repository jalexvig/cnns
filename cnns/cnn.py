import numpy as np

from cnns.utils import pad_image, convert_to_torch


def cnn2d(image: np.ndarray,
          filters: np.ndarray):
    """
    Vanilla convolutions.
    Args:
        image: (hi, wi, cin).
        filters: (hf, wf, cin, cout).

    Returns:
        (hi, wi, cout)
    """

    filter_len, image_padded = pad_image(filters, image)

    out = np.zeros([image.shape[0], image.shape[1], filters.shape[-1]])

    for idx_row in range(image.shape[0]):
        for idx_col in range(image.shape[1]):
            for idx_c_out in range(filters.shape[-1]):
                filter = filters[:, :, :, idx_c_out]
                patch = image_padded[idx_row: idx_row + filter_len, idx_col: idx_col + filter_len]
                out[idx_row, idx_col, idx_c_out] += (filter * patch).sum()

    return out


def cnn2d_torch(image: np.ndarray,
                filters: np.ndarray):

    from torch.nn import functional as F

    image_torch, filters_torch = convert_to_torch(image, filters)

    features_torch = F.conv2d(image_torch, filters_torch, padding=filters.shape[0] // 2)
    features_torch_ = features_torch.numpy()[0].transpose([2, 1, 0])

    return features_torch_


def cnn2d_tf(image: np.ndarray,
             filters: np.ndarray):

    import tensorflow as tf
    tf.enable_eager_execution()

    features_tf = tf.nn.conv2d(image[None], filters, strides=[1, 1, 1, 1], padding='SAME')

    return features_tf


if __name__ == '__main__':

    from cnns.settings import H_IMG, W_IMG, CIN, COUT, FILTER_LEN

    # Create inputs
    image = np.random.rand(H_IMG, W_IMG, CIN).astype(np.float32)
    filters = np.random.rand(FILTER_LEN, FILTER_LEN, CIN, COUT).astype(np.float32)

    # Convolve
    features_np = cnn2d(image, filters)

    # Compare with torch
    features_torch = cnn2d_torch(image, filters)
    print('Pytorch:', np.isclose(features_np, features_torch).all())

    # Compare with tensorflow
    features_tf = cnn2d_tf(image, filters)
    print('Tensorflow:', np.isclose(features_np, features_tf).all())
