import torch
import pickle
import os


def save_results(results, results_dir_path, results_path):
    os.makedirs(results_dir_path, exist_ok=True)
    with open(f'{results_dir_path}/{results_path}', 'wb') as file:
        pickle.dump(results, file)


def save_model(model, model_save_dir_path, model_path):
    os.makedirs(model_save_dir_path, exist_ok=True)
    torch.save(model, f'{model_save_dir_path}/{model_path}')


def load_model(model_path, device='cpu'):
    model = torch.load(model_path, map_location=device)  # .to(device)
    return model


def load_results(results_path):
    with open(results_path, 'rb') as file:
        results = pickle.load(file)
    return results
