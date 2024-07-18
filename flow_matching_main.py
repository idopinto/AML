import numpy as np

import create_data, engine, model_builder, utils
import torch
from matplotlib import pyplot as plt
import time
MODEL_PATH = 'checkpoints/flow_matching_models'
RESULTS_PATH = 'checkpoints/flow_matching_results'
PLOTS_DIR = 'plots/flow_matching_plots'
CHECKPOINTS_DIR = 'checkpoints'


def Q1_Loss(epochs, results, filename):
    pass


def Q2_flow_progression(model, seeds, n_samples, device, filename):
    pass


def Q3_point_trajectory(model, seed, n_layers, n_samples, device, filename):
    pass


def Q4_time_quantization(model, n_samples, device, filename):
    pass


def Q5_reversing_the_flow(model, seed, device, filename):
    pass


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
        "n_layers": 5,
        "in_features": 2,
        "out_features": 64,
        "delta_t": 0.001
    }
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    ####################################################################################################################
    ############################################## Initalization #######################################################
    ####################################################################################################################

    train_loader = utils.get_dataloader(config["train_size"], shuffle=True, batch_size=config["batch_size"], show=False)
    validation_loader = utils.get_dataloader(config["validation_size"], shuffle=False, batch_size=config["batch_size"],
                                             show=False)

    model = model_builder.UnconditionalFlowMatchingModel(n_layers=config["n_layers"],
                                               in_features=config["in_features"],
                                               out_features=config["out_features"],
                                               ).to(device)
    epochs = config["epochs"]
    criterion = model_builder.FlowMatchingCriterion()
    optimizer = config["optimizer"](model.parameters(), lr=config["learning_rate"])
    scheduler = config["lr_scheduler"](optimizer, T_max=epochs)

    ####################################################################################################################
    ############################################## Train Phase #########################################################
    ####################################################################################################################
    start_time = time.time()
    results = engine.train(model=model,
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

    Q2_flow_progression(model, seeds=(333, 666, 999), n_samples=1000, device=device, filename=filenames[1])

    Q3_point_trajectory(model,
                          seed=42,
                          n_layers=config['n_layers'],
                          n_samples=1000,
                          device=device,
                          filename=filenames[2])

    Q4_time_quantization(model,
                             n_samples=10,
                             device=device,
                             filename=filenames[3])
    Q5_reversing_the_flow(model, seed=42, device=device, filename=filenames[4])


if __name__ == '__main__':
    main()
