import numpy as np

import create_data, normalizing_flow_engine, model_builder, utils
import torch
from matplotlib import pyplot as plt
import time

MODEL_PATH = 'checkpoints/normalizing_flow_models'
RESULTS_PATH = 'checkpoints/normalizing_flow_results'
PLOTS_DIR = 'plots/normalizing_flows_plots'
CHECKPOINTS_DIR = 'checkpoints'


def pass_layer_by_layer(model, layer_outputs, inverse=False, device='cpu'):
    model = model.to(device)
    points = layer_outputs[0]
    model.eval()
    with torch.inference_mode():
        if not inverse:
            for i, layer in enumerate(model.interleaving_affine_coupling_layers):
                points, _ = layer(points)
                layer_outputs.append(points)
        else:
            for i, layer in enumerate(reversed(model.interleaving_affine_coupling_layers)):
                points, _ = layer.inverse(points)
                layer_outputs.append(points)
    return layer_outputs


def Q1_Loss(num_epochs, results, filename=""):
    """
    Present the validation loss over the training epochs.
    Additionally, plot the log-determinant and the prior log probability components of this loss
    in separate lines ate the same figure.
    :param filename:
    :param results: dictionary containing the validation loss over the training epochs and it's components
    :param num_epochs: number of training epochs
    :return:
    """
    epochs_range = range(num_epochs)
    plt.figure(figsize=(8, 8))
    plt.plot(epochs_range, results["validation_loss"], label="Validation Loss")
    plt.plot(epochs_range, results["total_log_det"], label="-log_det")
    plt.plot(epochs_range, results["total_log_prob"], label="-log_prob")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Validation Loss")
    plt.title("Validation Loss over epochs")
    plt.xticks(ticks=epochs_range)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


def Q2_sampling(model, seeds=(24, 42, 34), n_samples=1000, device='cpu', filename=""):
    samples = [utils.generate_samples(model, n_samples, seed=seed, device=device) for seed in seeds]
    utils.plot_samples(samples,
                       title="Generated Samples from Normalizing Flow Model",
                       sub_titles=[f"Samples from seed {seed}" for seed in seeds],
                       filename=filename)


def Q3_sampling_over_time(model, seed=42, n_layers=15, to_divide=5, n_samples=1000, device='cpu', filename=""):
    torch.manual_seed(seed)
    z = torch.randn(n_samples, 2, device=device)
    layer_outputs = pass_layer_by_layer(model, layer_outputs=[z], inverse=False, device=device)
    sub_outputs = [layer_outputs[i] for i in range(0, n_layers * 2 + 1, (n_layers * 2) // to_divide)]
    sub_titles = ["Prior samples"] + [f"Output after {i}/{to_divide} of the way" for i in range(1, to_divide + 1)]
    utils.plot_samples(sub_outputs,
                       sub_titles=sub_titles,
                       title="Sampling Over Time",
                       filename=filename)


def Q4_sampling_trajectories(model, n_samples=10, seed=42, device='cpu', filename=""):
    '''
    Sample points from your model and present the forward process of them layer by
    layer, as a trajectory in a 2D space. Color the points according to their time t.
    :param filename:
    :param device:
    :param n_samples:
    :param seed:
    :param model:
    :return:
    '''
    torch.manual_seed(seed)
    z = torch.randn(n_samples, 2, device=device)
    layer_outputs = pass_layer_by_layer(model, layer_outputs=[z], device=device)
    utils.plot_trajectories(n_samples=n_samples,
                            layer_outputs=layer_outputs,
                            title=f"Trajectories of Sampled Points Over Time",
                            filename=filename)


def compute_trajectories(model, data, title="", device="cpu", filename=""):
    layer_outputs = pass_layer_by_layer(model, inverse=True, layer_outputs=[data], device=device)
    utils.plot_trajectories(n_samples=data.shape[0],
                            layer_outputs=layer_outputs,
                            title=title,
                            filename=filename)


def Q5_probability_estimation(model, seed=42, device='cpu', filename=""):
    '''
    For 5 points of your choice, present the inverse process layer by layer, as a trajectory
    in a 2D space. Choose 3 points from inside the olympic logo and 2 outside of it. Color the points according to
    their time t
    :param filename:
    :param model:
    :param n_samples:
    :param seed:
    :param device:
    :return:
    '''
    torch.manual_seed(seed)
    in_points = torch.tensor([[-0.68536588, 1.04613213],
                              [-0.83722729, 1.48307046],
                              [0.71348099, -0.03766172]], device=device)

    out_points = torch.tensor([[-1.44, -1.44], [1.8, 1.8]], device=device)
    data = torch.cat((in_points, out_points))
    compute_trajectories(model, data,
                         title=f"Inverse trajectories of 5 data points over time",
                         device=device, filename=filename)
    model.eval()
    model = model.to(device)
    with torch.inference_mode():
        print("The log-probabilities of the inliers:")
        print("log probability: ", model.log_probability(data)[0][:3].tolist())
        print("prior: ", model.log_probability(data)[1][:3].tolist())
        print("The log-probabilities of the outliers:")
        print("log probability: ", model.log_probability(data)[0][3:].tolist())
        print("prior: ", model.log_probability(data)[1][3:].tolist())


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
    train_loader = utils.get_dataloader(n_points=config["train_size"], batch_size=config["batch_size"], show=False,
                                        shuffle=True)
    validation_loader = utils.get_dataloader(n_points=config["validation_size"], batch_size=config["batch_size"],
                                             show=False, shuffle=False)

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
    start_time = time.time()
    results = normalizing_flow_engine.train(model=model,
                                            train_loader=train_loader,
                                            val_loader=validation_loader,
                                            criterion=criterion,
                                            optimizer=optimizer,
                                            scheduler=scheduler,
                                            epochs=epochs,
                                            device=device,
                                            save=True,
                                            model_save_dir_path=MODEL_PATH,
                                            train_results_dir_path=RESULTS_PATH)
    end_time = time.time()
    total_time = end_time - start_time
    total_time_minutes = total_time // 60
    total_time_seconds = total_time % 60
    print(f"Total training time: {total_time_minutes} minutes and {total_time_seconds} seconds")
    ###################################################################################################################
    ######################################## Question Answering #######################################################
    ###################################################################################################################
    model_path = f"{MODEL_PATH}/nf_model_20_epochs.pth"
    results_path = f"{RESULTS_PATH}/nf_results_20_epochs.pkl"
    filenames = [f'{PLOTS_DIR}/Q1_loss_{epochs}_epochs.png',
                 f'{PLOTS_DIR}/Q2_random_samples.png',
                 f'{PLOTS_DIR}/Q3_sampling_over_time.png',
                 f'{PLOTS_DIR}/Q4_sampling_trajectories.png',
                 f'{PLOTS_DIR}/Q5_probability_estimation.png',
                 ]
    model, results = utils.load_model(model_path=model_path,
                                      results_path=results_path,
                                      device=device)
    Q1_Loss(epochs, results, filename=filenames[0])

    Q2_sampling(model, seeds=(333, 666, 999), n_samples=1000, device=device, filename=filenames[1])

    Q3_sampling_over_time(model,
                          seed=42,
                          n_layers=config['n_layers'],
                          n_samples=1000,
                          device=device,
                          filename=filenames[2])

    Q4_sampling_trajectories(model,
                             n_samples=10,
                             device=device,
                             filename=filenames[3])
    Q5_probability_estimation(model, seed=42, device=device, filename=filenames[4])


if __name__ == '__main__':
    main()
