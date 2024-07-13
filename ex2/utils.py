import os
import pickle

from create_data import *
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset


def get_dataloader(n_points=250000, batch_size=128, get_conditional=False, show=False, shuffle=True):
    # if get_conditional:
    #     data = create_olympic_rings(n_points, ring_thickness=0.25, verbose=show)
    #     data, labels = torch.tensor(data[0], dtype=torch.float32), torch.tensor(data[1])
    #
    #     return (DataLoader(TensorDataset(data, labels),
    #                        batch_size=batch_size, shuffle=shuffle), data[2])

    data = torch.tensor(create_unconditional_olympic_rings(n_points, ring_thickness=0.25, verbose=show),
                        dtype=torch.float32)
    return DataLoader(TensorDataset(data), batch_size=batch_size, shuffle=shuffle)


def plot_samples(samples_list, sub_headers, header, filename=None, color_map=None, labels=None, n_cols=3):
    n_plots = len(samples_list)
    num_cols = min(n_cols, n_plots)
    num_rows = int(np.ceil(n_plots / num_cols))
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows))
    if num_rows == 1: axs = [axs]
    if num_cols == 1: axs = [axs]
    i = 0
    for row in range(num_rows):
        while True:
            samples = samples_list[i]
            if labels != None and color_map:
                colors = [color_map[label.item()] for label in labels]
                axs[row][i % num_cols].scatter(samples[:, 0], samples[:, 1], c=colors, s=10)
            else:
                axs[row][i % num_cols].scatter(samples[:, 0], samples[:, 1], s=10)
            axs[row][i % num_cols].set_aspect('equal', adjustable='box')
            axs[row][i % num_cols].set_title(sub_headers[i])
            i += 1
            if i % num_cols == 0 or i == n_plots: break
        if i == n_plots: break

    plt.suptitle(header)
    if filename: plt.savefig(filename)
    plt.show()


def plot_samples(samples_list, sub_headers, header, filename=None, color_map=None, labels=None, n_cols=3):
    n_plots = len(samples_list)
    num_cols = min(n_cols, n_plots)
    num_rows = int(np.ceil(n_plots / num_cols))
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows))
    if num_rows == 1: axs = [axs]
    if num_cols == 1: axs = [axs]
    i = 0
    for row in range(num_rows):
        while True:
            samples = samples_list[i]
            if labels != None and color_map:
                colors = [color_map[label.item()] for label in labels]
                axs[row][i % num_cols].scatter(samples[:, 0], samples[:, 1], c=colors, s=10)
            else:
                axs[row][i % num_cols].scatter(samples[:, 0], samples[:, 1], s=10)
            axs[row][i % num_cols].set_aspect('equal', adjustable='box')
            axs[row][i % num_cols].set_title(sub_headers[i])
            i += 1
            if i % num_cols == 0 or i == n_plots: break
        if i == n_plots: break

    plt.suptitle(header)
    if filename: plt.savefig(filename)
    plt.show()

def load_model(model_path, results_path, device='cpu'):
    model = torch.load(model_path, map_location=device)  # .to(device)
    with open(results_path, 'rb') as file:
        results = pickle.load(file)
    return model, results


def save_model(model, results, model_save_dir_path, train_results_dir_path, model_path, results_path):
    os.makedirs(model_save_dir_path, exist_ok=True)
    torch.save(model, f'{model_save_dir_path}/{model_path}')
    os.makedirs(train_results_dir_path, exist_ok=True)
    with open(f'{train_results_dir_path}/{results_path}', 'wb') as file:
        pickle.dump(results, file)
