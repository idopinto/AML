import torch
from tqdm.auto import tqdm
import utils


def training_step(model, train_loader, criterion, optimizer, scheduler, device):
    """
    Perform a single training step, iterating over the training data, computing loss, and updating model weights.

    :param model: The model to be trained.
    :param train_loader: DataLoader containing the training data.
    :param criterion: Loss function to calculate the loss.
    :param optimizer: Optimizer for updating the model parameters.
    :param scheduler: Learning rate scheduler.
    :param device: The device to run the computations on (CPU or GPU).
    :return: The average training loss over the entire dataset.
    """
    model.train()
    training_loss = 0
    for batch in tqdm(train_loader):
        batch = batch[0].to(device)
        optimizer.zero_grad()
        log_prob, prior_log_prob, log_det_jacobian = model.log_probability(batch)
        loss = criterion(prior_log_prob, log_det_jacobian)
        training_loss += loss.item()
        loss.backward()
        optimizer.step()
    scheduler.step()

    return training_loss / len(train_loader)


def validation_step(model, val_loader, criterion, device):
    """
    Perform a single validation step, iterating over the validation data and computing loss without updating model weights.

    :param model: The model to be validated.
    :param val_loader: DataLoader containing the validation data.
    :param criterion: Loss function to calculate the loss.
    :param device: The device to run the computations on (CPU or GPU).
    :return: Tuple containing the average validation loss, total log probability, and total log determinant over the dataset.
    """
    model.eval()
    val_loss = total_log_det = total_log_prob = 0
    with torch.inference_mode():
        for batch in val_loader:
            batch = batch[0].to(device)
            log_prob, prior_log_prob, log_det_jacobian = model.log_probability(batch)
            loss = criterion(prior_log_prob, log_det_jacobian)
            total_log_det += -log_det_jacobian.mean().item()
            total_log_prob += -prior_log_prob.mean().item()
            val_loss += loss.item()
    return val_loss / len(val_loader), total_log_prob / len(val_loader), total_log_det / len(val_loader)


def train(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, device, save=True,
          model_save_dir_path=None, train_results_dir_path=None):
    """
    Train the model over a specified number of epochs, and optionally save the trained model and results.

    :param model: The model to be trained.
    :param train_loader: DataLoader containing the training data.
    :param val_loader: DataLoader containing the validation data.
    :param criterion: Loss function to calculate the loss.
    :param optimizer: Optimizer for updating the model parameters.
    :param scheduler: Learning rate scheduler.
    :param epochs: Number of training epochs.
    :param device: The device to run the computations on (CPU or GPU).
    :param save: Boolean flag indicating whether to save the trained model and results.
    :param model_save_dir_path: Directory path to save the trained model.
    :param train_results_dir_path: Directory path to save the training results.
    :return: Dictionary containing training and validation losses, as well as log determinant and log probability over epochs.
    """
    results = {
        "train_loss": [],
        "validation_loss": [],
        "total_log_det": [],
        "total_log_prob": []
    }
    for epoch in tqdm(range(epochs)):
        training_loss = training_step(model, train_loader, criterion, optimizer, scheduler, device)
        validation_loss, total_log_prob, total_log_det = validation_step(model, val_loader, criterion, device)
        results["train_loss"].append(training_loss)
        results["validation_loss"].append(validation_loss)
        results["total_log_det"].append(total_log_det)
        results["total_log_prob"].append(total_log_prob)
        if epoch % 1 == 0:
            print(f'Epoch {epoch + 1}/{epochs} | Training Loss: {training_loss} | Validation Loss: {validation_loss}')
    if save:
        model_path = f"fm_model_{epochs}_epochs.pth"
        results_path = f"fm_results_{epochs}_epochs.pkl"
        utils.save_model(model, results, model_save_dir_path, train_results_dir_path, model_path, results_path)

    return results
