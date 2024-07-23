import torch
from tqdm.auto import tqdm
import utils


def training_step(model, train_loader, criterion, optimizer, scheduler, device):
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
        "train_invariance_loss": [],
        "train_variance_loss": [],
        "train_covariance_loss": [],
        "validation_loss": [],
        "val_invariance_loss": [],
        "val_variance_loss": [],
        "val_covariance_loss": [],
    }
    for epoch in tqdm(range(epochs)):
        training_loss, train_components = training_step(model, train_loader, criterion, optimizer, scheduler, device)
        validation_loss, val_components = validation_step(model, val_loader, criterion, device)
        update_results(results, training_loss, train_components, validation_loss, val_components)
        if epoch % 1 == 0:
            print(f'Epoch {epoch + 1}/{epochs} | Training Loss: {training_loss} | Validation Loss: {validation_loss}')
    if save:
        model_path = f"{model.model_name}_model_{epochs}_epochs.pth"
        results_path = f"{model.model_name}_results_{epochs}_epochs.pkl"
        utils.save_model(model, model_save_dir_path, model_path)
        utils.save_results(results, train_results_dir_path, results_path)

    return results
def update_results(results, training_loss, train_components, val_loss, val_components):
    results["train_loss"].append(training_loss)
    results["train_invariance_loss"].append(train_components[0])
    results["train_variance_loss"].append(train_components[1])
    results["train_covariance_loss"].append(train_components[2])
    results["validation_loss"].append(training_loss)
    results["val_invariance_loss"].append(train_components[0])
    results["val_variance_loss"].append(train_components[1])
    results["val_covariance_loss"].append(train_components[2])
    return results
