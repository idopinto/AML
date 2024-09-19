import sklearn
from sklearn.metrics import silhouette_score

from utils import *
from data_setup import *
from config_class import *
from sklearn.cluster import KMeans


def plot_clusters(tsne_embeddings, kmeans_labels, actual_labels, centroids, title,
                  class_names, silhouette_score, filename=None):
    fig, axes = plt.subplots(1, 2, figsize=(25, 10))
    fig.suptitle(title, fontsize=16)

    # Plot KMeans clusters
    scatter_kmeans = axes[0].scatter(tsne_embeddings[:, 0],
                                     tsne_embeddings[:, 1],
                                     c=kmeans_labels,
                                     cmap='tab10',
                                     marker='o',
                                     s=3,
                                     alpha=0.6)

    axes[0].scatter(centroids[:, 0],
                    centroids[:, 1],
                    c='black',
                    marker='X',
                    s=30,
                    alpha=1,
                    label='Centroids')

    axes[0].set_title(f"T-SNE KMeans\nsilhouette score = {silhouette_score:.3f}")
    cbar_kmeans = plt.colorbar(scatter_kmeans,
                               ax=axes[0],
                               ticks=range(10))
    cbar_kmeans.set_label('Cluster Labels')

    # Plot actual classes
    scatter_actual = axes[1].scatter(tsne_embeddings[:, 0],
                                     tsne_embeddings[:, 1],
                                     c=actual_labels,
                                     cmap='tab10',
                                     marker='o',
                                     s=2,
                                     alpha=0.6)
    axes[1].set_title(f"T-SNE Original Labels\n")
    cbar_actual = plt.colorbar(scatter_actual,
                               ax=axes[1],
                               ticks=range(10))
    cbar_actual.ax.set_yticklabels(class_names)
    cbar_actual.set_label('Class Labels')

    if filename:
        plt.savefig(filename)
    plt.show()

def do_clustring(base_config, model_path, model_type, repr_path, plot_path, n_clusters=10):
    train_dataset = OriginalCIFAR10(root='./data', transform=test_transform, download=True)
    train_classes = np.array(train_dataset.dataset.targets)
    # Load model
    vicreg_model = get_loaded_model(config=base_config, model_path=model_path)
    # get the cifar10 trainset embeddings (50000, 128)
    trainset_embeddings = get_image_representations(vicreg_model, base_config.batch_size, filename=repr_path)
    # perform a dimensionality reduction to 2D using t-SNE
    reduced_embeddings = perform_pca_or_tsne(Y=trainset_embeddings,n_components=2,dim_reduction_type='tsne',perplexity=100)

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(reduced_embeddings)
    centroids = kmeans.cluster_centers_
    cluster_labels = kmeans.labels_
    silhouette_score_val = silhouette_score(reduced_embeddings, cluster_labels)
    plot_clusters(tsne_embeddings=reduced_embeddings,
                  kmeans_labels=cluster_labels,
                  actual_labels=train_classes,
                  centroids=centroids,
                  title=model_type,
                  class_names=CIFAR10_CLASSES,
                  silhouette_score=silhouette_score_val,
                  filename=plot_path)


def main():
    base_config = Config()
    do_clustring(base_config,
                 model_path=models_dir / "VICReg_model_30_epochs_.pth",
                 model_type="VICReg Model with generated neighbors",
                 repr_path=data_dir / "train_CIFAR10_Y_by_vicreg_30",
                 plot_path=plots_dir / f"p3q2_clustring_visualization_orig_vicreg.png",
                 )

    do_clustring(base_config,
                 model_path=models_dir / "VICReg_model_1_epochs_no_gen_neighbors.pth",
                 model_type="VICReg Model with no generated neighbors",
                 repr_path=data_dir / "train_CIFAR10_Y_by_vicreg_ngn_1_epoch",
                 plot_path=plots_dir / f"p3q2_clustring_visualization_orig_vicreg_ngn.png",
                 )

if __name__ == '__main__':
    main()