import pickle
import random
import faiss
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from tqdm import tqdm
import data_setup
from augmentations import train_transform, test_transform
from config_class import *

CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

################################## FUNCTIONS FOR PLOTS #################################
def save_collectors(collectors, collectors_path, verbose=False):
    collectors_path.parent.mkdir(parents=True, exist_ok=True)
    with open(collectors_path, 'wb') as file:
        pickle.dump(collectors, file)
    if verbose:
        print("Collectors saved!")


def save_model(model, model_path, verbose=False):
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    if verbose:
        print("Model checkpoint saved!")


def load_model(model, model_path, device='cpu', verbose=False):
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    if verbose:
        print(f"{model_path} loaded successfully!")
    return model


def load_collectors(collectors_path, verbose=False):
    with open(collectors_path, 'rb') as file:
        collectors = pickle.load(file)
    if verbose:
        print("Collectors loaded successfully!")
    return collectors


def freeze_encoder_only(encoder, verbose=False):
    for name, param in encoder.named_parameters():
        param.requires_grad = False
        if verbose:
            print(f"Layer: {name}, Requires Gradient: {param.requires_grad}")


def plot_losses(epochs, collectors_df, plot_path):
  fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(15, 4))
  epochs_range = range(epochs)
  titles = ["Total Loss", "Variance Loss", "Invariance Loss", "Covariance Loss"]
  xs = [["train_losses", "test_losses"],
        ["train_variance_losses", "test_variance_losses"],
        ["train_invariance_losses", "test_invariance_losses"],
        ["train_covariance_losses", "test_covariance_losses"]]
  for i in range(4):
      axs[i].plot(epochs_range,
                  collectors_df[xs[i][0]], label=xs[i][0], c="r")
      axs[i].plot(epochs_range,
                  collectors_df[xs[i][1]], label=xs[i][1], c="g")
      axs[i].set_title(titles[i])
      axs[i].set_xlabel("Epochs")
      axs[i].set_ylabel(titles[i])
      axs[i].legend()
  plt.tight_layout()
  plt.savefig(plot_path, format='png')
  plt.show()
  plt.close()

  print(f"Plot saved as '{plot_path}'")


def plot_dimensionality_reduction(Y_reduced_pca, Y_reduced_tsne, labels, plot_path):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))
    Y_reduced = [Y_reduced_pca, Y_reduced_tsne]
    titles = ['PCA of CIFAR10 Test Data', 't-SNE of CIFAR10 Test Data']
    for i in range(2):
        scatter = axs[i].scatter(Y_reduced[i][:, 0], Y_reduced[i][:, 1], c=labels, cmap='tab10', s=2)
        cbar = fig.colorbar(scatter, ax=axs[i], ticks=range(10))
        cbar.ax.set_yticklabels(CIFAR10_CLASSES)
        axs[i].set_xlabel('Component 1')
        axs[i].set_ylabel('Component 2')
        axs[i].set_title(titles[i])
    plt.savefig(plot_path, format='png')
    plt.show()
    plt.close()

    print(f"Plot saved as:'{plot_path}'")


def plot_image_nn(dataset, images, labels, nearest_neighbors, farthest_neighbors, k, title, plot_path):
    nrows = len(images)
    ncols = 2 * k + 1
    # print(nrows, ncols)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 20))
    fig.suptitle(title)
    for i in range(nrows):

        axs[i, 0].imshow(images[i])
        axs[i, 0].set_title(f"{CIFAR10_CLASSES[labels[i]]}")

        for j in range(1, len(nearest_neighbors[0])):
            img, _ = dataset[nearest_neighbors[i][j]]
            axs[i, j].imshow(img)
            axs[i, j].set_title(f"Nearest {j}")
        for j in range(1, len(nearest_neighbors[0])):
            img, _ = dataset[farthest_neighbors[i][j]]
            axs[i, j + k].imshow(img)
            axs[i, j + k].set_title(f"Farthest {j}")

    for ax in axs.ravel():
        ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(plot_path, format='png')
    plt.show()
    plt.close()

    print(f"Plot saved as '{plot_path}'")

def plot_images(random_images, random_labels):
    # ChatGPT
    # Plot the images
    plt.figure(figsize=(10, 5))
    for i, image in enumerate(random_images):
        plt.subplot(2, 5, i + 1)
        # Convert tensor to numpy array for plotting
        # image_np = image.numpy().transpose((1, 2, 0))  # Convert from CHW to HWC format
        plt.imshow(image)
        plt.title(CIFAR10_CLASSES[random_labels[i]])
        plt.axis('off')
    plt.show()

################## COMPUTATIONAL HELPER METHODS #######################
def perform_pca_or_tsne(Y, n_components, dim_reduction_type="pca"):
    if dim_reduction_type == "pca":
        # perform PCA
        print("Performing PCA on images representations..")
        pca = PCA(n_components=n_components)
        Y_reduced_pca = pca.fit_transform(Y)
        print("PCA finished..")
        return Y_reduced_pca

    if dim_reduction_type == "tsne":
        # Perform t-SNE
        print("Performing t-SNE on images representations..")
        # tsne = TSNE(n_components=n_components, random_state=0)
        tsne = TSNE(n_components=2, perplexity=150, n_iter=300)
        Y_reduced_tsne = tsne.fit_transform(Y)
        print("t-SNE finished..")
        return Y_reduced_tsne
    print("The dimensionality reduction type can be either PCA or t-SNE.")


def get_image_representations(model, batch_size=256, test_loader=None, filename=None):
    # load image representations file if already computed
    if filename is not None and filename.exists():
        print("Image representations already exist. Loading..")
        Y = torch.load(filename).astype(np.float32)
        print(f"{filename} successfully!")
        return Y

    Y, labels= [], []
    # get original trainset and train loader
    if test_loader is None:
        orig_trainset = data_setup.OriginalCIFAR10(root=data_dir, train=True, transform=test_transform)
        train_loader = DataLoader(orig_trainset, batch_size=batch_size, shuffle=False, num_workers=2)
        # compute the image representations and save them for later.
        print("Computing the image representations...")
        model.eval()
        with torch.inference_mode():
            for images, lbls in tqdm(train_loader):
                Y.append(model.encoder.encode(images.to(device)).cpu().numpy())
    else:
        model.eval()
        with torch.inference_mode():
            for images, lbls in tqdm(test_loader):
                Y.append(model.encoder.encode(images.to(device)).cpu().numpy())
                labels.append(lbls)
            labels = torch.cat(labels)
    Y = np.concatenate(Y, axis=0)
    if filename is not None:
        torch.save(Y, filename)
        print(f"Image representations saved in: {filename}")

    return Y if test_loader is None else (Y, labels)


def get_nearest_neighbors(Y, filename=Path(""), k=4):
    # Load nearest neighbors for image representations if already computed.
    if filename.exists():
        print(f"FAISS {k}-NN indices already exist. Loading...")
        indices = torch.load(filename).astype(np.float32)
        print("Loaded successfully!")
        return indices

    # else perform K-NN on the image representations using FAISS and save for later.
    # Create FAISS index
    index = faiss.IndexFlatL2(Y.shape[1])  # L2 distance index
    index.add(Y)  # Add representations to the index
    # Perform search for nearest neighbors
    distances, indices = index.search(Y, k)
    # Save the indices of the nearest neighbors
    if filename:
        torch.save(indices, filename)
        print(f"FAISS {k}-NN indices saved in: {filename}")
    return indices

def get_random_images(data_loader, num_images=10, num_classes=10):
    """
    Selects random images from each class in the dataset.
    """
    # ChatGPT
    random_images = []
    random_labels = []
    random_ind = []

    class_count = {i: 0 for i in range(num_classes)}
    while len(random_images) < num_images:
        idx = random.randint(0, len(data_loader) - 1)
        img, label = data_loader.dataset[idx]
        if class_count[label] < 1:
            random_images.append(img)
            random_ind.append(idx)
            random_labels.append(label)
            class_count[label] += 1
    # plot_images(random_images, random_labels)
    return random_images, random_labels, random_ind


def get_representations_and_nn_indices(model, batch_size, repr_path, indices_path, k=6):
    Y = get_image_representations(model, batch_size=batch_size, filename=repr_path)
    indices = get_nearest_neighbors(Y, filename=indices_path, k=k)
    return Y, indices

def find_neighbors(data, queries, k=6, return_farthest=False):
    # ChatGPT
    faiss_index = faiss.IndexFlatL2(data.shape[1])
    faiss_index.add(data)
    neighbors = []
    distances, indices = faiss_index.search(queries, len(data))
    for i in range(len(queries)):
        inds = indices[i][1:k + 1] if not return_farthest else indices[i][-k:]
        neighbors.append(inds)
    return neighbors
