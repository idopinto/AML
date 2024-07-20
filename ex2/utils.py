import os
import pickle

import matplotlib.pyplot as plt

from create_data import *
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from matplotlib.cm import get_cmap


def get_dataloader(n_points=250000, batch_size=128, get_conditional=False, show=False, shuffle=True):
    if get_conditional:
        sampled_points, labels, int_to_label = create_olympic_rings(n_points, ring_thickness=0.25, verbose=show)
        sampled_points, labels = torch.tensor(sampled_points, dtype=torch.float32), torch.tensor(labels)

        return DataLoader(TensorDataset(sampled_points, labels), batch_size=batch_size, shuffle=shuffle), int_to_label

    data = torch.tensor(create_unconditional_olympic_rings(n_points, ring_thickness=0.25, verbose=show),
                        dtype=torch.float32)
    return DataLoader(TensorDataset(data), batch_size=batch_size, shuffle=shuffle)


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


def plot_samples(samples_list, sub_titles, title, filename=None, color_map=None, labels=None, n_cols=3):
    n_plots = len(samples_list)
    num_cols = min(n_cols, n_plots)
    num_rows = int(np.ceil(n_plots / num_cols))
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows))
    if num_rows == 1: axs = [axs]
    if num_cols == 1: axs = [axs]
    i = 0
    for row in range(num_rows):
        while True:
            if torch.is_tensor(samples_list[i]):
                samples = samples_list[i].cpu().numpy()
            else:
                samples = samples_list[i]
            if labels != None and color_map:
                colors = [color_map[label.item()] for label in labels]
                axs[row][i % num_cols].scatter(samples[:, 0], samples[:, 1], c=colors, s=10)
            else:
                axs[row][i % num_cols].scatter(samples[:, 0], samples[:, 1], s=10)
            axs[row][i % num_cols].set_aspect('equal', adjustable='box')
            if sub_titles:
                axs[row][i % num_cols].set_title(sub_titles[i])
            i += 1
            if i % num_cols == 0 or i == n_plots: break
        if i == n_plots: break

    plt.suptitle(title)
    if filename: plt.savefig(filename)
    plt.show()


def generate_samples(model, num_samples, seed=None, device=None):
    if seed is not None:
        torch.manual_seed(seed)
    z = torch.randn(num_samples, 2, device=device)
    model.eval()
    with torch.inference_mode():
        model = model.to(device)
        samples = model(z)
    return samples.detach().cpu().numpy()


def plot_trajectories(n_samples=10, layer_outputs=None, title="", filename="",
                      color_map=None, labels=None, samples=None, samples_labels=None):
    plt.figure(figsize=(10, 8))

    edge_cmap = get_cmap('plasma')
    point_cmap = get_cmap('viridis')
    num_layers = len(layer_outputs)
    if color_map is not None and labels is not None:
        colors = [color_map[label.item()] for label in samples_labels]
        plt.scatter(samples[:, 0], samples[:, 1], c=colors, s=10)
    # Plot trajectories (lines) with consistent color for each sample using edge_cmap
    for i in range(n_samples):
        x_vals = [layer_outputs[j][i, 0].item() for j in range(num_layers)]
        y_vals = [layer_outputs[j][i, 1].item() for j in range(num_layers)]
        if color_map is not None and labels is not None:
            plt.plot(x_vals, y_vals, marker=None,
                     label=f"class {labels[i].item()}",
                     color=color_map[labels[i].item()],
                     linewidth=3)
        else:
            plt.plot(x_vals, y_vals, color=edge_cmap(i / n_samples), alpha=0.5, linestyle='-')

    # Overlay points with colors changing according to layer index using point_cmap
    for t, output in enumerate(layer_outputs):
        if torch.is_tensor(output):
            plt.scatter(output[:, 0].cpu().numpy(), output[:, 1].cpu().numpy(), color=point_cmap(t / num_layers))
        else:
            plt.scatter(output[:, 0], output[:, 1], c=point_cmap(t / num_layers))

    plt.title(title)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend(loc='best')
    sm_point = plt.cm.ScalarMappable(cmap=point_cmap)
    sm_point.set_array([])
    cbar_point = plt.colorbar(sm_point, ax=plt.gca(), pad=0.1)  # Specify ax=plt.gca() for current axis
    cbar_point.set_label('Time (Layer Index)')

    plt.grid(True)
    plt.savefig(filename)
    plt.show()


def plot_trajectories_over_samples(n_samples, trajectory, title="", filename="",
                                   color_map=None, labels=None, samples=None, samples_labels=None):
    fig, ax = plt.subplots(figsize=(10, 8))

    if color_map is not None and labels is not None:
        colors = [color_map[label.item()] for label in samples_labels]
        ax.scatter(samples[:, 0], samples[:, 1], c=colors, s=10)

    num_steps = len(trajectory)

    # Plot trajectories (lines) with consistent color for each sample using edge_cmap
    for i in range(n_samples):
        x_vals = [trajectory[j][i, 0].item() for j in range(num_steps)]
        y_vals = [trajectory[j][i, 1].item() for j in range(num_steps)]
        ax.plot(x_vals, y_vals, marker=None,
                label=f"class {labels[i].item()}",
                color=color_map[labels[i].item()],
                linewidth=3)
    # # Overlay points with colors changing according to layer index using point_cmap
    colors = plt.cm.viridis(np.linspace(0, 1, num_steps))

    for j in range(num_steps):
        x_coords_at_t = trajectory[j][:, 0]
        y_coords_at_t = trajectory[j][:, 1]
        ax.scatter(x_coords_at_t, y_coords_at_t, color=colors[j], marker='o', s=50, label=f'Time {j}' if j == 0 else "")

    # Create a color bar to show the progression of time
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=num_steps - 1))
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Time step')

    ax.set_title(title)
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.legend()
    ax.grid(True)
    if filename: plt.savefig(filename)
    plt.show()
