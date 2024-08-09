import torch
from tqdm.auto import tqdm
import utils


def training_step(model, train_loader, criterion, optimizer, scheduler, delta_t, device):
    """
    Perform a single training step for the model.

    :param model: The model to train.
    :param train_loader: DataLoader for the training dataset.
    :param criterion: Loss function.
    :param optimizer: Optimizer.
    :param scheduler: Learning rate scheduler.
    :param delta_t: Time step interval for sampling.
    :param device: Device to perform computations on (e.g., 'cpu' or 'cuda').
    :return: Average training loss for the step.
    """
    model.train()
    training_loss = 0
    time_steps = torch.arange(0, 1 + delta_t, delta_t, device=device)
    labels = None
    for batch in train_loader:
        optimizer.zero_grad()
        if model.model_name == "cfm":
            y_1, labels = batch[0].to(device), batch[1].to(device)
        else:
            y_1 = batch[0].to(device)
        # Sample random noise with the same size as batch
        y_0 = torch.randn(y_1.size()).to(device)
        # Sample time t uniformly from [0,1]
        t_indices = torch.randint(0, len(time_steps), (y_1.size(0),))  # Sample indices
        t = time_steps[t_indices].unsqueeze(1)  # Create tensor of sampled times
        y_t = t * y_1 + (1 - t) * y_0  # Compute intermediate state
        v_hat_t = model(y_t, t, labels) if model.model_name == "cfm" else model(y_t, t)
        loss = criterion(v_hat_t, y_0, y_1)
        training_loss += loss.item()
        loss.backward()
        optimizer.step()
    scheduler.step()

    return training_loss / len(train_loader)


def train(model, train_loader, criterion, optimizer, scheduler, epochs, delta_t, device, save=True,
          model_dir_path=None, results_dir_path=None):
    """
    Train the model for a specified number of epochs.

    :param model: The model to train.
    :param train_loader: DataLoader for the training dataset.
    :param criterion: Loss function.
    :param optimizer: Optimizer.
    :param scheduler: Learning rate scheduler.
    :param epochs: Number of epochs to train.
    :param delta_t: Time step interval for sampling.
    :param device: Device to perform computations on (e.g., 'cpu' or 'cuda').
    :param save: Whether to save the model and results.
    :param model_dir_path: Directory path to save the model.
    :param results_dir_path: Directory path to save the results.
    :return: Dictionary containing training loss history.
    """
    results = {
        "train_loss": [],
    }
    for epoch in tqdm(range(epochs)):
        training_loss = training_step(model, train_loader, criterion, optimizer, scheduler, delta_t, device)
        results["train_loss"].append(training_loss)
        if epoch % 5 == 0:
            print(f'Epoch {epoch + 1}/{epochs} | Training Loss: {training_loss}')
    if save:
        model_path = f"{model.model_name}_model_{epochs}_epochs.pth"
        results_path = f"{model.model_name}_results_{epochs}_epochs.pkl"
        utils.save_model(model, results, model_dir_path, results_dir_path, model_path, results_path)

    return results
