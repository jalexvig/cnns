import numpy as np

from cnns.utils import pad_image, convert_to_torch


def cnn2d_grouped(image: np.ndarray,
                  filters: np.ndarray,
                  groups: int = 1) -> np.ndarray:
    """
    Grouped convolutions. Can emulate depthwise convolutions but has bit more flexibility.
    Args:
        image: (hi, wi, cin).
        filters: (hf, wf, cin / groups, cout).
        groups: must divide cin and cout.

    Returns:
        (hi, wi, cout)
    """

    assert filters.shape[-1] % groups == 0
    assert image.shape[-1] == groups * filters.shape[-2]

    filter_len, image_padded = pad_image(filters, image)

    out = np.zeros([image.shape[0], image.shape[1], filters.shape[-1]])

    n_cin_group = image.shape[-1] // groups
    n_cout_group = filters.shape[-1] // groups

    for idx_group in range(groups):
        for idx_row in range(image.shape[0]):
            for idx_col in range(image.shape[1]):
                for idx_cout_group in range(n_cout_group):

                    idx_cout = n_cout_group * idx_group + idx_cout_group

                    filter = filters[:, :, :, idx_cout]
                    patch = image_padded[idx_row: idx_row + filter_len,
                            idx_col: idx_col + filter_len,
                            idx_group * n_cin_group: (idx_group + 1) * n_cin_group]

                    out[idx_row, idx_col, idx_cout] = (filter * patch).sum()

    return out


def cnn2d_grouped_torch(image: np.ndarray,
                        filters: np.ndarray,
                        groups: int = 1) -> np.ndarray:

    from torch.nn import functional as F

    image_torch, filters_torch = convert_to_torch(image, filters)

    features_torch = F.conv2d(image_torch, filters_torch, padding=filters.shape[0] // 2, groups=groups)
    features_torch_ = features_torch.numpy()[0].transpose([2, 1, 0])

    return features_torch_


if __name__ == '__main__':

    from cnns.settings import H_IMG, W_IMG, CIN, COUT, FILTER_LEN, GROUPS

    # Create inputs
    image = np.random.rand(H_IMG, W_IMG, CIN)
    filters = np.random.rand(FILTER_LEN, FILTER_LEN, CIN // GROUPS, COUT)

    # Convolve
    features_np = cnn2d_grouped(image, filters, GROUPS)

    # Compare with torch
    features_torch = cnn2d_grouped_torch(image, filters, GROUPS)
    print('Pytorch:', np.isclose(features_np, features_torch).all())
