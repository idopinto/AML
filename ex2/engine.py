import pickle

import torch
from torch import nn
from torch.optim import optimizer
from tqdm.auto import tqdm
import os
from pathlib import Path
import datetime
import utils


def training_step(model, train_loader, criterion, optimizer, scheduler, device):
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
        model_path = f"nf_model_{epochs}_epochs.pth"
        results_path = f"nf_results_{epochs}_epochs.pkl"
        utils.save_model(model, results, model_save_dir_path, train_results_dir_path, model_path, results_path)

    return results
