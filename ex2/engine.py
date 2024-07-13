import pickle

import torch
from torch import nn
from torch.optim import optimizer
from tqdm.auto import tqdm
import os
import datetime

def training_step(model, train_loader, criterion, optimizer, scheduler, device):
    '''
    This is the training step function.
    :param model:
    :param train_loader:
    :param criterion:
    :param optimizer:
    :param scheduler:
    :param device:
    :return:
    '''
    model.train()
    training_loss = 0
    for batch in train_loader:
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
    This is the validation step function.
    :param model:
    :param val_loader:
    :param criterion:
    :param device:
    :return:
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
          model_save_path=None, train_results_path=None):
    """
    This function is used to train the model.
    :param model: The model to be trained.
    :param train_loader: The training data loader.
    :param val_loader: The validation data loader.
    :param criterion: The loss function.
    :param optimizer: The optimizer.
    :param scheduler: The learning rate scheduler.
    :param epochs: The number of epochs to train the model.
    :param device: The device to train the model on.
    :param save: Whether to save the model weights.
    :param model_save_path: The path to save the model weights.
    :param train_results_path: Path to save the training results.
    :return: training loss and validation loss.
    """
    results = {
        "train_loss": [],
        "validation_loss": [],
        "total_log_det": [],
        "total_log_prob": []
    }
    for epoch in tqdm(range(epochs)):
        training_loss = training_step(model, train_loader, criterion, optimizer, scheduler, device)
        validation_loss, total_log_det, total_log_prob = validation_step(model, val_loader, criterion, device)
        results["train_loss"].append(training_loss)
        results["validation_loss"].append(validation_loss)
        results["total_log_det"].append(total_log_det)
        results["total_log_prob"].append(total_log_prob)
        if epoch % 1 == 0:
            print(f'Epoch {epoch + 1}/{epochs} | Training Loss: {training_loss} | Validation Loss: {validation_loss}')
    if save:
        # from datetime import datetime
        # current_date_time = datetime.now()
        os.makedirs(model_save_path, exist_ok=True)
        torch.save(model, f'{model_save_path}/nf_{epochs}_epochs.pth')
        os.makedirs(train_results_path, exist_ok=True)
        with open(f'{train_results_path}/nf_{epochs}_epochs.pkl', 'wb') as file:
            pickle.dump(results, file)

    return results
