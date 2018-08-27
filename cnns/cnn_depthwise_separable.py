import numpy as np

from cnns import cnn2d, cnn2d_depthwise
from cnns.utils import convert_to_torch


def cnn2d_depthwise_separable(image: np.ndarray,
                              filters_depthwise: np.ndarray,
                              filters_pointwise: np.ndarray):
    """
    Depthwise separable convolutions.
    Args:
        image: (hi, wi, cin).
        filters_depthwise: (hfd, wfd, cin, cmul).
        filters_pointwise: (1, 1, cin * cmul, cout).

    Returns:
        (hi, wi, cout)
    """

    if len(filters_pointwise.shape) == 2:
        filters_pointwise = filters_pointwise[None, None]

    out_depthwise = cnn2d_depthwise(image, filters_depthwise)

    out = cnn2d(out_depthwise, filters_pointwise)

    return out


def cnn2d_depthwise_separable_torch(image: np.ndarray,
                                    filters_depthwise: np.ndarray,
                                    filters_pointwise: np.ndarray):

    from torch.nn import functional as F

    image_torch, filters_depthwise_torch = convert_to_torch(image, filters_depthwise)

    df, _, cin, cmul = filters_depthwise.shape
    filters_depthwise_torch = filters_depthwise_torch.transpose(0, 1).contiguous()
    filters_depthwise_torch = filters_depthwise_torch.view(cin * cmul, 1, df, df)

    features_depthwise_torch = F.conv2d(image_torch, filters_depthwise_torch, padding=df // 2, groups=cin)

    _, filters_pointwise_torch = convert_to_torch(image, filters_pointwise)

    features_pointwise_torch = F.conv2d(features_depthwise_torch, filters_pointwise_torch)
    features_pointwise_torch_ = features_pointwise_torch.numpy()[0].transpose([2, 1, 0])

    return features_pointwise_torch_


def cnn2d_depthwise_separable_tf(image: np.ndarray,
                                 filters_depthwise: np.ndarray,
                                 filters_pointwise: np.ndarray):

    import tensorflow as tf
    tf.enable_eager_execution()

    features_tf = tf.nn.separable_conv2d(image[None], filters_depthwise, filters_pointwise, strides=[1, 1, 1, 1], padding='SAME')

    return features_tf


if __name__ == '__main__':

    from cnns.settings import H_IMG, W_IMG, CIN, COUT, FILTER_LEN, CMUL

    # Create inputs
    image = np.random.rand(H_IMG, W_IMG, CIN).astype(np.float32)
    filters_depthwise = np.random.rand(FILTER_LEN, FILTER_LEN, CIN, CMUL).astype(np.float32)
    filters_pointwise = np.random.rand(1, 1, CIN * CMUL, COUT).astype(np.float32)

    # Convolve
    features_np = cnn2d_depthwise_separable(image, filters_depthwise, filters_pointwise)

    # Compare to pytorch
    features_torch = cnn2d_depthwise_separable_torch(image, filters_depthwise, filters_pointwise)
    print('Pytorch:', np.isclose(features_np, features_torch).all())

    # Compare to tensorflow
    features_tf = cnn2d_depthwise_separable_tf(image, filters_depthwise, filters_pointwise)
    print('Tensorflow:', np.isclose(features_tf[0], features_np).all())
