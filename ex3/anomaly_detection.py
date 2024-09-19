
from models import VICRegModel
from utils import *
from data_setup import *
from config_class import *
import faiss
from sklearn.metrics import roc_curve, auc


def get_knn_density(vicreg_model,trainset_embeddings, test_loader, k=3):
    testset_embeddings, labels = get_image_representations(vicreg_model, batch_size=256, test_loader=test_loader)
    index = faiss.IndexFlatL2(trainset_embeddings.shape[1])
    index.add(trainset_embeddings)
    distances, _= index.search(testset_embeddings,k) #
    knn_density = np.mean(distances[:, 1:], axis=1)  # Exclude the first distance (self distance)
    return knn_density, labels

def plot_roc_curve(y_true, y_score, model_type =None, filename=None):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(10,10))
    ax.plot(fpr, tpr, color='blue', label='ROC curve (area = %0.2f)' % roc_auc)
    ax.plot([0, 1], [0, 1], color='navy', linestyle='--')
    ax.set_xlim([0.0, 1.0])

    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'(ROC) Receiver operating characteristic of {model_type}; AUC={roc_auc:0.2f}')
    ax.legend(loc="lower right")
    if filename is not None:
        plt.savefig(filename)
    plt.show()


def do_anomaly_detection(base_config,model_path, model_type, repr_path, plot1_path,plot2_path):
    # Load model
    vicreg_model = get_loaded_model(config=base_config, model_path=model_path)
    # get test set (with anomalies) and test loader
    testset = CIFAR10_MIXED_WITH_MNIST(root=data_dir,transform=test_transform, download=True)
    test_loader = DataLoader(testset, batch_size=base_config.batch_size, shuffle=False)
    # get test set and test loader with no transformations for qualitative evaluation
    testset_no_aug = CIFAR10_MIXED_WITH_MNIST(root=data_dir,transform=None, download=True)
    test_loader_no_aug = DataLoader(testset_no_aug, batch_size=base_config.batch_size, shuffle=False)
    # get the normal data embeddings (50000, 128)
    trainset_embeddings = get_image_representations(vicreg_model,base_config.batch_size, filename=repr_path)
    # get knn density and labels for plotting the roc curve (20000, 1)
    knn_density, labels = get_knn_density(vicreg_model, trainset_embeddings, test_loader,k=3)
    # q2 plot the roc curve with auc.
    plot_roc_curve(y_true=labels, y_score=knn_density,model_type=model_type, filename=plot1_path)
    # plot the 7 most anomalous samples according to the model.
    qualitative_evaluation(knn_density, test_loader_no_aug, m=7, filename=plot2_path)


def qualitative_evaluation(knn_density, test_loader, m=7, filename=None):
    # chatgpt
    # Get the indices of the top m images with the highest density values
    indices = np.argsort(knn_density)[-m:]

    # Create a subplot grid with 1 row and m columns
    fig, axes = plt.subplots(1, m, figsize=(15, 3))

    # If there's only one subplot, `axes` is not an array, so we need to handle it
    if m == 1:
        axes = [axes]

    # Iterate over the indices and plot the corresponding images
    for i, idx in enumerate(indices):
        img, label = test_loader.dataset[idx]
        axes[i].imshow(img)
        axes[i].set_title(f"Label: {label}")
        axes[i].axis('off')  # Turn off axis

    # Save the figure if a filename is provided
    if filename is not None:
        plt.savefig(filename)

    # Show the plot
    plt.show()

def main():
    base_config = Config()
    do_anomaly_detection(base_config,
                         model_path=models_dir / "VICReg_model_30_epochs_.pth",
                         model_type = "VICReg Model with generated neighbors",
                         repr_path = data_dir / "train_CIFAR10_Y_by_vicreg_30",
                         plot1_path = plots_dir/f"p2q2_roc_curve_orig_vicreg.png",
                         plot2_path = plots_dir/f"p2q3_qualitative_evaluation_orig_vicreg.png"
                         )

    do_anomaly_detection(base_config,
                         model_path=models_dir / "VICReg_model_1_epochs_no_gen_neighbors.pth",
                         model_type = "VICReg Model with no generated neighbors",
                         repr_path = data_dir/"train_CIFAR10_Y_by_vicreg_ngn_1_epoch",
                         plot1_path = plots_dir/f"p2q2_roc_curve_orig_vicreg_ngn.png",
                         plot2_path = plots_dir / f"p2q3_qualitative_evaluation_vicreg_ngn.png"
                         )

if __name__ == '__main__':
    main()