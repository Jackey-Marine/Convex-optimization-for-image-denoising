'''Plotting functions. '''

import pickle
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


# ordered_algos = ['quad', 'TV', 'nlm', 'wnnm', 'EMC', 'VMF', 'MTM', 'GF']
# algo_colors = {'quad':'cornflowerblue', 'TV':'goldenrod', 'nlm':'pink', 'wnnm':'tomato', 'EMC': 'aquamarine', 'VMF': 'slategrey', 'MTM': 'teal', 'GF':'mediumorchid'}

ordered_algos = ['quad', 'TV', 'nlm', 'wnnm', 'VMF', 'MTM', 'GF']
algo_colors = {'quad':'cornflowerblue', 'TV':'goldenrod', 'nlm':'pink', 'wnnm':'tomato', 'VMF': 'slategrey', 'MTM': 'teal', 'GF':'aquamarine'}


def plot_experiment(all_results, noise_type):
    '''Plots the results for every hyperparameter variation.

    Args:
        all_results (dict): The dictionary containing the data.
        noise_type (str): The type of noise, either 'gaussian' or 'poisson'.
    '''

    fig = plt.figure()
    ax = fig.add_subplot(111)

    cert_data = defaultdict(list)
    cert_data_std = defaultdict(list)
    labels = []

    if noise_type == 'gaussian':
        hyperparameters = [0.01, 0.025, 0.05]
    elif noise_type == 'poisson':
        hyperparameters = [50, 20, 10]
    elif noise_type == 'salt-pepper':
        hyperparameters = [0.05]

    for param_idx, param in enumerate(hyperparameters):
        if noise_type == 'gaussian':
            labels.append(r'$\sigma$' + f'={param}')
        elif noise_type == 'poisson':
            labels.append(f'Photons={param}')
        elif noise_type == 'salt-pepper':
            labels.append(f'Amount={param}')

        for algo in ordered_algos:
            raw_data = [all_results[algo][im][param_idx] for im in all_results[algo].keys() if im != 'noisy']
            cert_data[algo].append(np.mean(raw_data))
            cert_data_std[algo].append(np.std(raw_data))

    num_bars = len(ordered_algos)
    width = 1/(num_bars+1)
    x = np.arange(len(labels))

    bars = {}

    for idx, algo in enumerate(ordered_algos):
        if algo == 'quad':
            algo_name = 'Quadratic'
        else:
            algo_name = algo.upper()
        position = ((num_bars-1)/2.0 - idx)*width
        bars[algo] = ax.bar(x - position, cert_data[algo], yerr=cert_data_std[algo], width=width, label=f'{algo_name}', color=algo_colors[algo], capsize=10)

    plt.xticks(range(3), labels, fontsize=25)
    ax.legend(fontsize=25, loc='lower left')
    ax.set_ylabel('PSNR', fontsize=25)

    for algo in ordered_algos:
        rounded_labels = []
        for l in cert_data[algo]:
            if l > 100:
                rounded_labels.append(int(l))
            elif l > 1:
                rounded_labels.append(round(l, 1))
            else:
                rounded_labels.append(round(l, 2))
        ax.bar_label(bars[algo], labels=rounded_labels, padding=3, fontsize=20)

    fig.tight_layout()

    ax.set_ylim(ymin=0)
    ax.yaxis.grid(True)
    plt.show()


if __name__ == '__main__':
    noise_name = 'gaussian'
    with open(f'./results/{noise_name}/PSNR_results.pkl', 'rb') as f:
        PSNR_data = pickle.load(f)
    with open(f'./results/{noise_name}/time_results.pkl', 'rb') as f:
        time_data = pickle.load(f)

    plot_experiment(PSNR_data, noise_name)
