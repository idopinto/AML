import numpy as np

import create_data, engine, model_builder, utils
import torch
from matplotlib import pyplot as plt
import time
from matplotlib.cm import get_cmap

MODEL_PATH = "checkpoints/normalizing_flow_models"
RESULTS_PATH = 'checkpoints/normalizing_flow_results'
PLOTS_DIR = "plots/normalizing_flows_plots"
CHECKPOINTS_DIR = "checkpoints"


def Q1_Loss(num_epochs, results):
    """
    Present the validation loss over the training epochs.
    Additionally, plot the log-determinant and the prior log probability components of this loss
    in separate lines ate the same figure.
    :param results: dictionary containing the validation loss over the training epochs and it's components
    :param num_epochs: number of training epochs
    :return:
    """
    validation_loss = results["validation_loss"]
    total_log_det = results["total_log_det"]
    total_log_prob = results["total_log_prob"]
    epochs_range = range(num_epochs)

    plt.figure(figsize=(8, 8))
    plt.plot(epochs_range, validation_loss, label="Validation Loss")
    plt.plot(epochs_range, total_log_det, label="-log_det")
    plt.plot(epochs_range, total_log_prob, label="-log_prob")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Validation Loss")
    plt.title("Validation Loss over epochs")
    plt.xticks(ticks=range(num_epochs))  # Set x-axis ticks to integer values of epochs
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/normalizing_flow_loss_{num_epochs}_epochs.png')
    plt.show()
    # plt.close()


def Q2_sampling(model, seeds=(24, 42, 34), n_samples=1000):
    samples = [utils.generate_samples(model, n_samples, seed=seed) for seed in seeds]
    utils.plot_samples(samples,
                       header="Generated Samples from Normalizing Flow Model",
                       sub_headers=[f"Samples from seed {seed}" for seed in seeds],
                       filename=f"{PLOTS_DIR}/Q2_random_samples.png")


def pass_layer_by_layer(model, layer_outputs, inverse=False, device='cpu'):
    model = model.to(device)
    z = layer_outputs[0]
    model.eval()
    with torch.inference_mode():
        for i, layer in enumerate(model.interleaving_affine_coupling_layers):
            z, _ = layer(z) if inverse else layer.inverse(z)
            layer_outputs.append(z)
    return layer_outputs


def Q3_sampling_over_time(model, seed=66, n_layers=15, to_divide=5, n_samples=1000, device='cpu'):
    if seed is not None:
        torch.manual_seed(seed)
    z = torch.randn(n_samples, 2, device=device)
    layer_outputs = pass_layer_by_layer(model, layer_outputs=[z], device=device)
    sub_outputs = [layer_outputs[i] for i in range(0, n_layers * 2 + 1, (n_layers * 2) // to_divide)]
    utils.plot_samples(sub_outputs,
                       sub_headers=["Prior samples"] +
                                   [f"Output after {i}/{to_divide} of the way"
                                    for i in range(1, to_divide + 1)],
                       header="Sampling Over Time",
                       filename=f"{PLOTS_DIR}/Q3_sampling_over_time.png")


# def f(x_coords, y_coords):
#     num_trajectories, points_per_trajectory = x_coords.shape
#     points_colors = np.linspace(0, 1, points_per_trajectory)
#     line_colors = np.linspace(0, 1, num_trajectories)
#     for
def Q4_sampling_trajectories(model, n_samples=10, seed=66, device='cpu'):
    '''
    Sample points from your model and present the forward process of them layer by
    layer, as a trajectory in a 2D space. Color the points according to their time t.
    :param model:
    :return:
    '''
    torch.manual_seed(seed)
    z = torch.randn(n_samples, 2, device=device)
    layer_outputs = pass_layer_by_layer(model, layer_outputs=[z], device=device)
    plot_trajectories(n_samples=n_samples, layer_outputs=layer_outputs)

def Q5_probability_estimation(model, seed=66, device='cpu'):
    '''
    For 5 points of your choice, present the inverse process layer by layer, as a trajectory
    in a 2D space. Choose 3 points from inside the olympic logo and 2 outside of it. Color the points according to
    their time t
    :param model:
    :param n_samples:
    :param seed:
    :param device:
    :return:
    '''
    torch.manual_seed(seed)
    points = torch.tensor([[-1, -2],  # extreme outlier
                           [1, -1.5],  # mild outlier
                           [0.5, -1.8],  # 1-intersection inlier
                           [0, -1],  # 2-intersection inlier
                           [3, 0]])  # 3-intersection inlier
    # print(points.shape)
    layer_outputs = pass_layer_by_layer(model,inverse=True, layer_outputs=[points], device=device)
    plot_trajectories(n_samples=points.shape[0], layer_outputs=layer_outputs)
    model.eval()
    with torch.inference_mode():
        log_probs, _, _ = model.log_probability(points)
        print(log_probs.shape)
        for i in range(points.shape[0]):
            print(f"point {i}: {log_probs[i]}")



def plot_trajectories(n_samples=1, layer_outputs=None):
    plt.figure(figsize=(10, 8))

    edge_cmap = get_cmap('plasma')
    point_cmap = get_cmap('viridis')
    num_layers = len(layer_outputs)

    # Plot trajectories (lines) with consistent color for each sample using edge_cmap
    for i in range(n_samples):
        x_vals = [layer_outputs[j][i, 0].item() for j in range(num_layers)]
        y_vals = [layer_outputs[j][i, 1].item() for j in range(num_layers)]
        plt.plot(x_vals, y_vals, color=edge_cmap(i / n_samples), alpha=0.5, linestyle='-')

    # Overlay points with colors changing according to layer index using point_cmap
    for t, output in enumerate(layer_outputs):
        # if t >= 30:
        plt.scatter(output[:, 0].cpu().numpy(), output[:, 1].cpu().numpy(),
                    color=point_cmap(t / num_layers), label=f'Time {t}' if t == 0 else None)

    plt.title('Trajectories of Sampled Points Over Time')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend(loc='best')
    sm_point = plt.cm.ScalarMappable(cmap=point_cmap)
    sm_point.set_array([])
    cbar_point = plt.colorbar(sm_point, pad=0.1)
    cbar_point.set_label('Time (Layer Index)')

    plt.grid(True)
    plt.savefig(f'{PLOTS_DIR}/Q4_sampling_trajectories.png')
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


def main():
    ####################################################################################################################
    ########################################## Configuration Setup #####################################################
    ####################################################################################################################
    config = {
        "optimizer": torch.optim.Adam,
        "learning_rate": 1e-3,
        "train_size": 250000,
        "validation_size": 50000,
        "epochs": 20,
        "batch_size": 128,
        "lr_scheduler": torch.optim.lr_scheduler.CosineAnnealingLR,
        "n_layers": 15,
        "in_features": 2,
        "out_features": 8
    }
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    ####################################################################################################################
    ############################################## Initalization #######################################################
    ####################################################################################################################
    train_loader = utils.get_dataloader(config["train_size"], shuffle=True, batch_size=config["batch_size"], show=False)
    validation_loader = utils.get_dataloader(config["validation_size"], shuffle=False, batch_size=config["batch_size"],
                                             show=False)

    model = model_builder.NormalizingFlowModel(n_layers=config["n_layers"],
                                               in_features=config["in_features"],
                                               out_features=config["out_features"],
                                               ).to(device)
    epochs = config["epochs"]
    criterion = model_builder.NormalizingFlowCriterion()
    optimizer = config["optimizer"](model.parameters(), lr=config["learning_rate"])
    scheduler = config["lr_scheduler"](optimizer, T_max=epochs)
    ####################################################################################################################
    ############################################## Train Phase #########################################################
    ####################################################################################################################
    # start_time = time.time()
    # results = engine.train(model=model,
    #                        train_loader=train_loader,
    #                        val_loader=validation_loader,
    #                        criterion=criterion,
    #                        optimizer=optimizer,
    #                        scheduler=scheduler,
    #                        epochs=epochs,
    #                        device=device,
    #                        save=True,
    #                        model_save_dir_path=MODEL_PATH,
    #                        train_results_dir_path=RESULTS_PATH)
    # end_time = time.time()
    # total_time = end_time - start_time
    # print(f"Total training time: {total_time} seconds")
    ###################################################################################################################
    ######################################## Question Answering #######################################################
    ###################################################################################################################
    model, results = utils.load_model(model_path=f"{MODEL_PATH}/nf_20_epochs.pth",
                                      results_path=f"{RESULTS_PATH}/nf_20_epochs.pkl", device=device)
    # Q1_Loss(epochs, results)
    # Q2_sampling(model, seeds=(541, 66, 86), n_samples=1000)
    # Q3_sampling_over_time(model, n_layers=15, n_samples=1000, device=device)
    # Q4_sampling_trajectories(model, n_samples=10, device=device)
    # data = create_data.create_unconditional_olympic_rings(n_points=50000)
    Q5_probability_estimation(model)


if __name__ == '__main__':
    main()
