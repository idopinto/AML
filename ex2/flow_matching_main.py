import numpy as np
import create_data, flow_matching_engine, model_builder, utils
import torch
from matplotlib import pyplot as plt
import time

# Paths for saving models, results, and plots
MODEL_PATH = 'checkpoints/flow_matching_models'
RESULTS_PATH = 'checkpoints/flow_matching_results'
PLOTS_DIR = 'plots/flow_matching_plots'
CHECKPOINTS_DIR = 'checkpoints'
uc_filenames = [f'{PLOTS_DIR}/part_2_q1_loss_20_epochs.png',
                f'{PLOTS_DIR}/part_2_q2_flow_progression.png',
                f'{PLOTS_DIR}/part_2_q3_point_trajectory.png',
                f'{PLOTS_DIR}/part_2_q4_time_quantization.png',
                f'{PLOTS_DIR}/part_2_q5_reversing_the_flow_1.png',
                f'{PLOTS_DIR}/part_2_q5_reversing_the_flow_2.png',
                ]
c_filenames = [
    f'{PLOTS_DIR}/part_3_q1_plotting_the_input.png',
    f'{PLOTS_DIR}/part_3_q2_a_point_from_each_class.png',
    f'{PLOTS_DIR}/part_3_q3_point_sampling.png',
]


def propagate_through_time(model, z, labels=None, initial_t=0, target_t=1, delta_t=0.001, get_trajectory=False):
    """
    Propagates the input through time using the model.

    :param model: The model to use for propagation.
    :param z: Input tensor to propagate.
    :param labels: Labels for conditional models, if any.
    :param initial_t: Starting time.
    :param target_t: Ending time.
    :param delta_t: Time step size.
    :param get_trajectory: If True, returns the trajectory of the input.
    :return: Final output after propagation or (final output, trajectory) if get_trajectory is True.
    """
    trajectory = [z.detach().cpu().numpy()] if get_trajectory else None
    model.eval()
    with torch.inference_mode():
        if initial_t < target_t:
            time_steps = torch.arange(initial_t + delta_t, target_t + delta_t, delta_t, device=z.device)
            time_steps = time_steps.repeat(z.shape[0], 1)
            for t in time_steps.T:
                z = (z + model(z, t.unsqueeze(1)) * delta_t) if labels is None else (
                        z + model(z, t.unsqueeze(1), labels) * delta_t)
                if get_trajectory:
                    trajectory.append(z.detach().cpu().numpy())
        else:
            time_steps = torch.arange(initial_t - delta_t, target_t - delta_t, -delta_t, device=z.device)
            time_steps = time_steps.repeat(z.shape[0], 1)
            for t in time_steps.T:
                z = (z - model(z, t.unsqueeze(1)) * delta_t) if labels is None else (
                        z - model(z, t.unsqueeze(1), labels) * delta_t)
                if get_trajectory:
                    trajectory.append(z.detach().cpu().numpy())

    return (z, trajectory) if get_trajectory else z


def Part_2_Q1_Loss(num_epochs, results, filename):
    """
    Plots training loss over epochs.

    :param num_epochs: Number of epochs.
    :param results: Dictionary containing training results.
    :param filename: Path to save the plot.
    """
    epochs_range = range(num_epochs)
    plt.figure(figsize=(8, 8))
    plt.plot(epochs_range, results["train_loss"], label="Training Loss")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Training Loss")
    plt.title("Training Loss over epochs")
    plt.xticks(ticks=epochs_range)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


def Part_2_Q2_flow_progression(model, n_samples, time_steps, delta_t=0.001, device='cpu', filename=""):
    """
    Plots the progression of samples through time.

    :param model: The model to use for propagation.
    :param n_samples: Number of samples to propagate.
    :param time_steps: List of time steps to visualize.
    :param delta_t: Time step size.
    :param device: Device to perform computations on.
    :param filename: Path to save the plot.
    """
    z = torch.randn(n_samples, 2, device=device)
    outputs = []
    for t in time_steps:
        y_t = propagate_through_time(model=model,
                                     z=z,
                                     initial_t=0,
                                     target_t=t,
                                     delta_t=delta_t,
                                     get_trajectory=False)
        outputs.append(y_t)

    sub_titles = [f"Output in time {t:.1f}" for t in time_steps]
    utils.plot_samples(outputs,
                       sub_titles=sub_titles,
                       title="Flow Progression",
                       filename=filename)


def Part_2_Q3_point_trajectory(model, n_samples, device, filename):
    """
    Plots the trajectory of points through time.

    :param model: The model to use for propagation.
    :param n_samples: Number of samples.
    :param device: Device to perform computations on.
    :param filename: Path to save the plot.
    """
    z = torch.randn(n_samples, 2, device=device)
    y, trajectory = propagate_through_time(model, z, initial_t=0, target_t=1, delta_t=0.001, get_trajectory=True)
    utils.plot_trajectories(n_samples=n_samples,
                            layer_outputs=trajectory,
                            title=f"10 Points trajectory through time",
                            filename=filename)


def Part_2_Q4_time_quantization(model, n_samples, delta_ts, device='cpu', filename=''):
    """
    Plots samples generated with different time step sizes.

    :param model: The model to use for generation.
    :param n_samples: Number of samples.
    :param delta_ts: List of time step sizes.
    :param device: Device to perform computations on.
    :param filename: Path to save the plot.
    """
    z = torch.randn(n_samples, 2, device=device)
    outputs = []
    for delta_t in delta_ts:
        y_1 = propagate_through_time(model, z, initial_t=0, target_t=1, delta_t=delta_t, get_trajectory=False)
        outputs.append(y_1)

    sub_titles = [f"Generated Samples with delta_t = {delta_t}" for delta_t in delta_ts]
    utils.plot_samples(outputs,
                       sub_titles=sub_titles,
                       title="Generated Samples from Unconditional Flow Matching model using different delta_t",
                       filename=filename)


def Part_2_Q5_reversing_the_flow(model, device='cpu', filenames=()):
    """
    Plots the trajectories of points reversed through the flow.

    :param model: The model to use for propagation.
    :param device: Device to perform computations on.
    :param filenames: List of filenames to save the plots.
    """
    in_points = torch.tensor([[-0.68536588, 1.04613213],
                              [-0.83722729, 1.48307046],
                              [0.71348099, -0.03766172]], device=device)

    out_points = torch.tensor([[-1.44, -1.44], [1.8, 1.8]], device=device)

    data = torch.cat((in_points, out_points))
    # print(f"original data:{data}")
    z, inverse_trajectory = propagate_through_time(model, data, initial_t=1, target_t=0, delta_t=0.001,
                                                   get_trajectory=True)

    y, trajectory = propagate_through_time(model, z, initial_t=0, target_t=1, delta_t=0.001, get_trajectory=True)
    # print(f"Transformed data:{y}")
    utils.plot_trajectories(n_samples=5,
                            layer_outputs=inverse_trajectory,
                            title=f"5 Points inverse trajectory through time",
                            filename=filenames[0])
    utils.plot_trajectories(n_samples=5,
                            layer_outputs=trajectory,
                            title=f"Forward pass over the inverse points through time",
                            filename=filenames[1])


def Part_3_Q1_plotting_the_input(n_points=3000, filename=''):
    """
    Plots samples of the training data.

    :param n_points: Number of points to plot.
    :param filename: Path to save the plot.
    """
    train_loader, color_map = utils.get_dataloader(n_points=n_points, batch_size=n_points, get_conditional=True)
    points, labels = next(iter(train_loader))
    utils.plot_samples(samples_list=[points],
                       sub_titles=None,
                       title="Samples of Training Data",
                       filename=filename, color_map=color_map, labels=labels, n_cols=3)


def Part_3_Q2_a_point_from_each_class(model, device='cpu', filename=''):
    """
    Plots the trajectory of a point from each class and validates that it reaches its class region.

    :param model: The model to use for propagation.
    :param device: Device to perform computations on.
    :param filename: Path to save the plot.
    """
    train_loader, color_map = utils.get_dataloader(n_points=10000, batch_size=10000, get_conditional=True, show=False)
    samples, samples_labels = next(iter(train_loader))

    labels = torch.tensor([0, 1, 2, 3, 4], device=device)
    points = torch.randn(labels.shape[0], 2, device=device)
    y, trajectory = propagate_through_time(model, points, labels, initial_t=0, target_t=1, delta_t=0.001,
                                           get_trajectory=True)
    utils.plot_trajectories(n_samples=5, layer_outputs=trajectory, title="Trajectories of Sampled Points over time",
                            filename=filename,
                            color_map=color_map,
                            labels=labels, samples=samples, samples_labels=samples_labels)


def Part_3_Q3_sampling(model, color_map, device='cpu', filename=''):
    """
    Plots samples generated for each class over different time steps.

    :param model: The model to use for generation.
    :param color_map: Color map for the classes.
    :param device: Device to perform computations on.
    :param filename: Path to save the plot.
    """
    labels = torch.tensor([0, 1, 2, 3, 4] * 600, device=device)
    z = torch.randn(labels.shape[0], 2, device=device)
    outputs = []
    time_steps = [0, 0.2, 0.4, 0.6, 0.8, 1]
    for t in time_steps:
        y_t = propagate_through_time(model=model,
                                     z=z,
                                     labels=labels,
                                     initial_t=0,
                                     target_t=t,
                                     delta_t=0.001,
                                     get_trajectory=False)
        outputs.append(y_t)

    sub_titles = [f"Output in time {t:.1f}" for t in time_steps]
    title = 'Generated samples colored by their classes'
    utils.plot_samples(samples_list=outputs, sub_titles=sub_titles, title=title, filename=filename, color_map=color_map,
                       labels=labels)


def run_part_2(model, results, epochs, device, filenames):
    """
    Runs Part 2 of the experiments.

    :param model: The model to use.
    :param results: Training results.
    :param epochs: Number of epochs.
    :param device: Device to perform computations on.
    :param filenames: List of filenames to save plots.
    """
    Part_2_Q1_Loss(epochs, results, filename=filenames[0])
    Part_2_Q2_flow_progression(model, n_samples=1000,
                               time_steps=torch.tensor([0, 0.2, 0.4, 0.6, 0.8, 1.0], device=device),
                               device=device, filename=filenames[1])
    Part_2_Q3_point_trajectory(model,
                               n_samples=10,
                               device=device,
                               filename=filenames[2])
    Part_2_Q4_time_quantization(model,
                                n_samples=1000,
                                delta_ts=[0.002, 0.02, 0.05, 0.1, 0.2, 1/3],
                                device=device,
                                filename=filenames[3])
    Part_2_Q5_reversing_the_flow(model, device=device,
                                 filenames=[filenames[4],
                                            filenames[5]])


def run_part_3(model, color_map, device, c_filenames):
    """
    Runs Part 3 of the experiments.

    :param model: The model to use.
    :param color_map: Color map for the classes.
    :param device: Device to perform computations on.
    :param c_filenames: List of filenames to save plots.
    """
    Part_3_Q1_plotting_the_input(n_points=3000, filename=c_filenames[0])
    Part_3_Q2_a_point_from_each_class(model, device, c_filenames[1])
    Part_3_Q3_sampling(model, color_map, device=device, filename=c_filenames[2])


def run_bonus(model, device='cpu'):
    """
    Runs the bonus experiment to check the reversibility of the flow.

    :param model: The model to use.
    :param device: Device to perform computations on.
    """
    y = torch.tensor([[4, 5]], device=device)
    print(f"y: {y}")
    z = propagate_through_time(model, y, initial_t=1, target_t=0, delta_t=0.00001)
    print(f"z ----> y : {z}")
    y = propagate_through_time(model, z, initial_t=0, target_t=1, delta_t=0.00001)
    print(f"y ----> z : {y}")


def train_unconditional_flow_matching_model(config, train_mode=True, device='cpu'):
    """
    Trains an unconditional flow matching model.

    :param config: Configuration dictionary for training.
    :param train_mode: Whether to train the model.
    :param device: Device to perform computations on.
    :return: The trained model and training results.
    """
    train_loader = utils.get_dataloader(config["train_size"], batch_size=config["batch_size"],
                                        get_conditional=False, show=False, shuffle=True)
    model = model_builder.UnconditionalFlowMatchingModel(
        in_features=config["in_features"],
        out_features=config["out_features"]
    ).to(device)
    epochs = config["epochs"]
    criterion = model_builder.FlowMatchingCriterion()
    optimizer = config["optimizer"](model.parameters(), lr=config["learning_rate"])
    scheduler = config["lr_scheduler"](optimizer, T_max=epochs)
    delta_t = config["delta_t"]
    results = {}
    if train_mode:
        print(f"Training unconditional flow matching model for {epochs} epochs")
        start_time = time.time()
        results = flow_matching_engine.train(model=model,
                                             train_loader=train_loader,
                                             criterion=criterion,
                                             optimizer=optimizer,
                                             scheduler=scheduler,
                                             epochs=epochs,
                                             delta_t=delta_t,
                                             device=device,
                                             save=True,
                                             model_dir_path=MODEL_PATH,
                                             results_dir_path=RESULTS_PATH)
        end_time = time.time()
        total_time = end_time - start_time
        total_time_minutes = total_time // 60
        total_time_seconds = total_time % 60
        print(f"Total training time: {int(total_time_minutes)} minutes and {total_time_seconds:.2f} seconds")
    return model, results


def train_conditional_flow_matching_model(config, train_mode=True, device='cpu'):
    """
    Trains a conditional flow matching model.

    :param config: Configuration dictionary for training.
    :param train_mode: Whether to train the model.
    :param device: Device to perform computations on.
    :return: The trained model, training results, and color map.
    """
    train_loader, color_map = utils.get_dataloader(config["train_size"], batch_size=config["batch_size"],
                                                   get_conditional=True, show=False, shuffle=True)
    num_classes = len(color_map)
    model = model_builder.ConditionalFlowMatchingModel(in_features=config["in_features"],
                                                       out_features=config["out_features"],
                                                       num_classes=num_classes,
                                                       embedding_dim=config["embedding_dim"]).to(device)
    epochs = config["epochs"]
    criterion = model_builder.FlowMatchingCriterion()
    optimizer = config["optimizer"](model.parameters(), lr=config["learning_rate"])
    scheduler = config["lr_scheduler"](optimizer, T_max=epochs)
    delta_t = config["delta_t"]
    results = {}
    if train_mode:
        print(f"Training conditional flow matching model for {epochs} epochs")
        start_time = time.time()
        results = flow_matching_engine.train(model=model,
                                             train_loader=train_loader,
                                             criterion=criterion,
                                             optimizer=optimizer,
                                             scheduler=scheduler,
                                             epochs=epochs,
                                             delta_t=delta_t,
                                             device=device,
                                             save=True,
                                             model_dir_path=MODEL_PATH,
                                             results_dir_path=RESULTS_PATH)
        end_time = time.time()
        total_time = end_time - start_time
        total_time_minutes = total_time // 60
        total_time_seconds = total_time % 60
        print(f"Total training time: {int(total_time_minutes)} minutes and {total_time_seconds:.2f} seconds")
    return model, results, color_map


def main():
    config = {
        "optimizer": torch.optim.Adam,
        "learning_rate": 1e-3,
        "train_size": 250000,
        "validation_size": 50000,
        "epochs": 20,
        "batch_size": 128,
        "lr_scheduler": torch.optim.lr_scheduler.CosineAnnealingLR,
        "n_layers": 5,
        "in_features": 2,
        "out_features": 64,
        "delta_t": 0.001,
        "seed": 42,
        "conditional": False,
        "embedding_dim": 10
    }
    seed = config["seed"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    torch.manual_seed(seed)
    fm_model, fm_results = train_unconditional_flow_matching_model(config, train_mode=False, device=device)
    cfm_model, cfm_results, color_map = train_conditional_flow_matching_model(config, train_mode=False, device=device)

    # Load previously trained models and results
    fm_model, fm_results = utils.load_model(model_path=f"{MODEL_PATH}/fm_model_20_epochs.pth",
                                            results_path=f"{RESULTS_PATH}/fm_results_20_epochs.pkl",
                                            device=device)

    cfm_model, cfm_results = utils.load_model(model_path=f"{MODEL_PATH}/cfm_model_20_epochs.pth",
                                              results_path=f"{RESULTS_PATH}/cfm_results_20_epochs.pkl",
                                              device=device)
    run_part_2(fm_model, fm_results, config["epochs"], device, uc_filenames)
    # run_part_3(cfm_model, color_map, device, c_filenames)
    # run_bonus(fm_model, device)


if __name__ == '__main__':
    main()
