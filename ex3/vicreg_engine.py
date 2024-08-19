import torch
from tqdm.auto import tqdm
import utils
def training_step(model, train_loader, criterion, optimizer, device):
    model.train()
    collectors = {
        "training_loss": 0,
        "train_invariance_loss": 0,
        "train_variance_loss": 0 ,
        "train_covariance_loss": 0
    }
    for aug_images1, aug_images2, _ in tqdm(train_loader):
        aug_images1 = aug_images1.to(device)
        aug_images2 = aug_images2.to(device)
        optimizer.zero_grad()
        # forward pass
        _, Z = model(aug_images1)
        _, Z_tag = model(aug_images2)
        # calculate loss and it's components
        loss, components = criterion(Z, Z_tag)
        loss.backward()
        optimizer.step()

        # update statistics
        collectors["training_loss"] += loss.item()
        collectors["train_invariance_loss"] += components[0].item()
        collectors["train_variance_loss"] += components[1].item()
        collectors["train_covariance_loss"] += components[2].item()


    # normalize statstics
    collectors["training_loss"] /= len(train_loader)
    collectors["train_invariance_loss"] /= len(train_loader)
    collectors["train_variance_loss"] /= len(train_loader)
    collectors["train_covariance_loss"] /= len(train_loader)

    return collectors

def test_step(model, test_loader, criterion, device):
    model.eval()
    collectors = {
        "test_loss": 0,
        "test_invariance_loss": 0,
        "test_variance_loss": 0 ,
        "test_covariance_loss": 0
    }
    with torch.inference_mode():
        for aug_images1, aug_images2, _ in tqdm(test_loader):
            aug_images1 = aug_images1.to(device)
            aug_images2 = aug_images2.to(device)
            _ , Z = model(aug_images1)
            _ , Z_tag = model(aug_images2)
            loss, components = criterion(Z, Z_tag)
            collectors["test_loss"] += loss.item()
            collectors["test_invariance_loss"] += components[0].item()
            collectors["test_variance_loss"] += components[1].item()
            collectors["test_covariance_loss"] += components[2].item()

    collectors["test_loss"] /= len(test_loader)
    collectors["test_invariance_loss"] /= len(test_loader)
    collectors["test_variance_loss"] /= len(test_loader)
    collectors["test_covariance_loss"] /= len(test_loader)
    return collectors

def train(model, train_loader, test_loader, criterion, optimizer, epochs, device, save=True,
          model_path=None,collectors_path=None):
    collectors = {
        "train_losses": [],
        "train_invariance_losses": [],
        "train_variance_losses": [],
        "train_covariance_losses": [],
        "test_losses": [],
        "test_invariance_losses": [],
        "test_variance_losses": [],
        "test_covariance_losses": [],
    }
    for epoch in tqdm(range(epochs)):
        train_collectors = training_step(model, train_loader, criterion, optimizer,device)
        test_collectors = test_step(model, test_loader, criterion, device)
        update_collectors(collectors,train_collectors, test_collectors)
        if epoch % 1 == 0:
            print(f'Epoch {epoch + 1}/{epochs} | Training Loss: {train_collectors["training_loss"]}| Test Loss: {test_collectors["test_loss"]}')
    if save:
        # model_path = f"VICReg_model_{epochs}_epochs.pth"
        # collectors_path = f"VICReg_model_collectors_{epochs}_epochs.pkl"
        utils.save_model(model, model_path)
        utils.save_collectors(collectors,collectors_path)

    return collectors


def update_collectors(collectors, train_collectors, test_collectors):
    collectors["train_losses"].append(train_collectors["training_loss"])
    collectors["train_invariance_losses"].append(train_collectors["train_invariance_loss"])
    collectors["train_variance_losses"].append(train_collectors["train_variance_loss"])
    collectors["train_covariance_losses"].append(train_collectors["train_covariance_loss"])
    collectors["test_losses"].append(test_collectors["test_loss"])
    collectors["test_invariance_losses"].append(test_collectors["test_invariance_loss"])
    collectors["test_variance_losses"].append(test_collectors["test_variance_loss"])
    collectors["test_covariance_losses"].append(test_collectors["test_covariance_loss"])
    return collectors


def train_classifier(model, train_loader, criterion, optimizer, epochs, device, save=True,
                     model_path=None, collectors_path=None):
    for epoch in tqdm(range(epochs)):
        model.train()
        for images, labels in tqdm(train_loader):
          optimizer.zero_grad()
          images, labels = images.to(device), labels.to(device)
          logits = model(images)
          loss = criterion(logits, labels)
          loss.backward()
          optimizer.step()
    if save:
        utils.save_model(model, model_path)
    return model

def test_classifier(model, test_loader, device):
  correct = 0
  total = 0
  model.eval()
  with torch.inference_mode():
    for images, labels in tqdm(test_loader):
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            probs = torch.nn.functional.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            correct += torch.sum(preds == labels).item()
            total += labels.size(0)
  return correct / total