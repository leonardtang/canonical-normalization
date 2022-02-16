import matplotlib.pyplot as plt
import math
import numpy as np
from collections import defaultdict
from data_helper import load_data
from skimage.feature import greycomatrix, greycoprops


def pixel_correlation(img, distances, angles):
    """
    Calculate and return the correlation for the given distances and angles
    """

    co_matrices = greycomatrix(image=img, distances=distances, angles=angles)
    corrs = defaultdict(list)
    full_corr = greycoprops(co_matrices, 'correlation')
    for d_idx, distance in enumerate(distances):
        for a_idx, angle in enumerate(angles):
            corr_d_a = full_corr[d_idx][a_idx]
            corrs[angle].append(corr_d_a)

    return corrs

def normalize(center, neighbors, gamma, sigma, n):
    """
    :param center: center of receptive field
    :param neighbors: pixels to normalize across, including the center
    :param gamma: freeform parameter
    :param sigma: freeform parameter
    :param n: freeform parameter
    :return: normalized image
    """
    num = math.pow(center, n)
    den = math.pow(sigma, n) + np.sum(np.power(neighbors, n))
    return gamma * num / den


def decorrelate_img(img, neighbor_distance=1):
    """
    Decorrelate pixels in image via normalization
    """
    rows, cols = img.shape
    decorrelated = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            low_row = max(0, i - neighbor_distance)
            up_row = min(i + neighbor_distance + 1, rows)
            low_col = max(0, j - neighbor_distance)
            up_col = min(j + neighbor_distance + 1, cols)
            neighbors = img[low_row: up_row, low_col: up_col].flatten()
            center = img[i, j]

            new_val = normalize(center, neighbors, gamma=4, sigma=2, n=2)
            decorrelated[i][j] = new_val

    return decorrelated


if __name__ == "__main__":
    img_list = load_data()
    test_img = img_list[0]
    og_corrs = pixel_correlation(test_img,
                                 distances=[1, 2, 3, 4, 5],
                                 angles=[i * np.pi / 4 for i in range(4)])

    plt.style.use('ggplot')

    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
    for i, (angle, corr_list) in enumerate(og_corrs.items()):
        ax1.plot([distance + 1 for distance in range(len(corr_list))], corr_list, label=f"{i + 1}pi/4")

    ax1.legend()
    ax1.set_title("Pre-Normalization")
    ax1.set_xlabel("Offset Magnitude")
    ax1.set_ylabel("Correlation")

    decorrelated = decorrelate_img(test_img)
    rounded = np.rint(decorrelated).astype(np.uint8)
    post_corrs = pixel_correlation(rounded,
                                   distances=[1, 2, 3, 4, 5],
                                   angles=[i * np.pi / 4 for i in range(4)])

    for i, (angle, corr_list) in enumerate(post_corrs.items()):
        ax2.plot([distance + 1 for distance in range(len(corr_list))], corr_list, label=f"{i + 1}pi/4")

    ax2.legend()
    ax2.set_title("Post-Normalization")
    ax2.set_xlabel("Offset Magnitude")
    plt.savefig("decorrelate.png")
    plt.show()
