import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from statistics import NormalDist

def albrecht_contrast(stimulus_contrast, r_max, c_50, n):
    """
    :param stimulus_contrast: stimulus contrast, a percentage measuring grating contrast
    :param r_max:
    :param c_50:
    :param n:
    :return: Albrecht and Hamilton function of stimulus contrast
    """
    return r_max * (stimulus_contrast ** n) / (c_50 ** n + stimulus_contrast ** n)


def neuron_response(theta, f_contrast, sigma=65):
    """
    :param theta: angle of grating rotation, measured in degrees
    :param sigma: standard deviation of population response; default 65 is from Busse et al (originally Rust et al 2006)
    :param f_contrast: function of stimulus contrast, i.e. Albrecht Contrast function
    Model a population (list) of V1 neuronal responses as Gaussian * Albrecht Contrast
    """
    population_idx = np.random.normal(theta, sigma, 100)
    # Plug into PDF to get y-values
    gaussian = NormalDist(mu=theta, sigma=sigma)
    population_values = [gaussian.pdf(idx) * f_contrast for idx in population_idx]
    grid_idx = np.linspace(norm.ppf(0.01, loc=theta, scale=sigma),
                           norm.ppf(0.99, loc=theta, scale=sigma),
                           100)
    curve = norm.pdf(grid_idx, loc=theta, scale=sigma)
    return population_idx, np.asarray(population_values), curve


def normalize(inputs, gamma=1, sigma=1, n=1):
    """
    Normalization equation (equation 10) from Carandini and Heeger.
    Normalize across all neuronal responses, in particular not pixels.
    """
    summed_inputs = np.sum(np.power(inputs, n))
    for i, input_j in enumerate(inputs):
        inputs[i] = gamma * (np.power(input_j, n)) / (np.power(sigma, n) + summed_inputs)

    return inputs


def winner_takes_all():

    labels = []
    idx_and_mixtures = []

    for h_contrast in [0, 0.12, 0.25, 0.5]:
        for v_contrast in [0, 0.5]:

            f_h_contrast = albrecht_contrast(h_contrast, r_max=1, c_50=1, n=1)
            f_v_contrast = albrecht_contrast(v_contrast, r_max=1, c_50=1, n=1)

            h_index, h_response, h_curve = neuron_response(0, f_h_contrast)
            v_index, v_response, v_curve = neuron_response(90, f_v_contrast)

            normed = normalize([h_response, v_response], gamma=2, sigma=2, n=2)

            full_index = np.concatenate((h_index, v_index))
            full_response = np.concatenate((normed[0], normed[1]))
            idx_and_mixtures.append((full_index, full_response))
            labels.append((h_contrast, v_contrast))

    return labels, idx_and_mixtures

def main():

    labels, mixed_responses = winner_takes_all()
    fig, axes = plt.subplots(2, 4, figsize=(12, 5), sharex='all', sharey='all')
    axes = axes.ravel()

    for i, (idx, response) in enumerate(mixed_responses):
        axes[i].scatter(idx, response, s=1, c="maroon")
        h_c, y_c = labels[i]
        axes[i].set_title(f"0D-{h_c}C : 90D-{y_c}C")

    plt.suptitle("Winner-Takes-All via Normalization")
    plt.tight_layout()
    plt.savefig("winner.png")
    plt.show()

if __name__ == "__main__":
    main()
