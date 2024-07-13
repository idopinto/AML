import numpy as np

import create_data, engine, model_builder, utils
import torch
from matplotlib import pyplot as plt
import time

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


def generate_samples(model, num_samples, seed=None, return_original=False, device=None):
    '''
    :param model:
    :param num_samples:
    :param seed:
    :param return_original:
    :param device:
    :return:
    '''
    if seed is not None:
        torch.manual_seed(seed)
    z = torch.randn(num_samples, 2, device=device)
    model.eval()
    with torch.inference_mode():
        model = model.to(device)
        samples = model(z)
    if return_original:
        return samples.detach().cpu().numpy(), z.detach().cpu().numpy()
    return samples.detach().cpu().numpy()


def Q2_sampling(model, seeds=(24, 42, 34), n_samples=1000):
    samples = [generate_samples(model, n_samples, seed=seed) for seed in seeds]
    utils.plot_samples(samples,
                       header="Generated Samples from Normalizing Flow Model",
                       sub_headers=[f"Samples from seed {seed}" for seed in seeds],
                       filename=f"{PLOTS_DIR}/Q2_random_samples.png")


def Q3_sampling_over_time(model, n_layers=5, n_samples=1000):
    outputs = generate_samples(model, n_samples, get_trajectory=True)
    sub_outputs = np.array(outputs)[[i for i in range(0, len(outputs), len(outputs) // n_layers)]]
    utils.plot_samples(sub_outputs,
                       sub_headers=["Prior samples"] +
                                   [f"Output after {i}/{n_layers} of the way"
                                    for i in range(1, n_layers + 1)],
                       header="Sampling Over Time",
                       filename=f"{PLOTS_DIR}/Q3_sampling_over_time.png")


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
    Q1_Loss(epochs, results)
    Q2_sampling(model, seeds=(541, 66, 86), n_samples=1000)
    # Q3_sampling_over_time(model)


if __name__ == '__main__':
    main()
