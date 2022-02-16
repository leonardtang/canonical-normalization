import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
from data_helper import load_light_data

def weber_contrast(img, border=1):
    """
    Calculate Weber contrast of given `img` using background/neighborhood of thickness `border`.
    Intermediary: image obtained by subtracting the intensity at each location by the mean of background
    :return contrast map given by C = (I - I_m) / I_m
    """

    img_m = cv2.blur(img, ksize=(border * 2 + 1, border * 2 + 1))
    C = (img - img_m) / img_m
    return C

def adaptive_normalize(inputs, weights, alphas, gamma, sigma, n, m, p, beta):
    """
    :param inputs: set of sensory inputs (neuron's proxy receptive field)
    :param weights: coefficients of linear combination of sensory inputs
    :param alphas: weights to define a suppressive field surrounding a given neuron
    :param gamma: freeform parameter
    :param sigma: freeform parameter
    :param n: freeform parameter
    :param m: freeform parameter
    :param p: freeform parameter
    :param beta: accounts for spontaneous activity of input
    :return: normalized response of neuron
    Normalization equation (equation 12) from Carandini and Heeger.
    Normalize across a stimulus, i.e. a single pixel value.
    """
    num = (np.sum(np.dot(weights, inputs))) + beta
    den = math.pow(sigma, n) + np.power(np.sum(
        np.dot(np.asarray(alphas), np.power(np.asarray(inputs), m))
    ), p)

    response = gamma * num / den
    return response


def simple_normalization(background_intensity, center, sigma, n):
    return ((center + background_intensity) ** n) / (sigma ** n + (center + background_intensity) ** n)


def light_adaptation(img, neighbor_distance=1, gamma=4, sigma=2, n=2, m=1, p=2, beta=1):
    print("Performing light adaptation...")
    rows, cols = img.shape
    adapted = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            # To change if different receptive field is desired
            low_row = max(0, i - neighbor_distance)
            up_row = min(i + neighbor_distance + 1, rows)
            low_col = max(0, j - neighbor_distance)
            up_col = min(j + neighbor_distance + 1, cols)
            inputs = img[low_row: up_row, low_col: up_col].flatten()

            # Change summation/suppression field weights to be more interesting
            weights = [1 for i in inputs]
            weights[0] = 0.3
            weights[-1] = 0.3
            alphas = [1 for i in inputs]
            alphas[0] = 0.1
            alphas[-1] = 0.1
            new_val = adaptive_normalize(inputs, weights=weights, alphas=alphas,
                                         gamma=gamma, sigma=sigma, n=n, m=m, p=p, beta=beta)

            adapted[i][j] = new_val

    return adapted


def plot_contrast_estimate():
    print("Calculating contrast estimate...")
    base_image, brightened, darkened = load_light_data()
    base_adapt = light_adaptation(base_image)
    base_weber = weber_contrast(base_image)
    brightened_adapt = light_adaptation(brightened)
    brightened_weber = weber_contrast(brightened)
    darkened_adapt = light_adaptation(darkened)
    darkened_weber = weber_contrast(darkened)

    plt.gray()
    fig, axes = plt.subplots(3, 2, figsize=(5, 10), sharex='all', sharey='all')
    axes = axes.ravel()
    axes[0].imshow(base_adapt)
    axes[0].set_title("Base Adapt")
    axes[1].imshow(base_weber)
    axes[1].set_title("Base Weber")
    axes[2].imshow(brightened_adapt)
    axes[2].set_title("Brightened Adapt")
    axes[3].imshow(brightened_weber)
    axes[3].set_title("Brightened Weber")
    axes[4].imshow(darkened_adapt)
    axes[4].set_title("Darkened Adapt")
    axes[5].imshow(darkened_weber)
    axes[5].set_title("Darkened Weber")
    plt.suptitle("Neural Estimate of Contrast")
    plt.tight_layout()
    plt.savefig("contrast.png")
    plt.show()


def response_shift():
    print("Simulating contrast estimate...")
    grid = np.linspace(-10, 10, num=100)
    dark = [(point, simple_normalization(background_intensity=0, center=point, sigma=1, n=5)) for point in grid if
            1 > simple_normalization(background_intensity=0, center=point, sigma=1, n=5) >= 0]
    base = [(point, simple_normalization(background_intensity=2, center=point, sigma=1, n=5)) for point in grid if
            1 > simple_normalization(background_intensity=2, center=point, sigma=1, n=5) >= 0]
    bright = [(point, simple_normalization(background_intensity=4, center=point, sigma=1, n=5)) for point in grid if
              0 <= simple_normalization(background_intensity=4, center=point, sigma=1, n=5) < 1]

    dark_idx = [e[0] for e in dark]
    dark_val = [e[1] for e in dark]
    base_idx = [e[0] for e in base]
    base_val = [e[1] for e in base]
    bright_idx = [e[0] for e in bright]
    bright_val = [e[1] for e in bright]
    plt.plot(bright_idx, bright_val, label="Base")
    plt.plot(base_idx, base_val, label="Bright")
    plt.plot(dark_idx, dark_val, label="Dark")
    plt.legend()
    plt.suptitle("Response Shift via Light Adaptation")
    plt.savefig("response_shift.png")
    plt.show()


def main():
    plot_contrast_estimate()
    response_shift()


if __name__ == "__main__":
    main()
