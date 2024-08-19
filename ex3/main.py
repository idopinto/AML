from torch.utils.data import DataLoader
import pandas as pd
from augmentations import train_transform, test_transform
import vicreg_engine, utils
from utils import *
from models import VICRegModel, VICRegLinearProbing
from criterions import VICRegCriterion
from config_class import *
# more imports
from pathlib import Path


def q1_training(config, overrides=None, train=False, plot_path=Path(""), suffix="", show_plot=True):
  if overrides:
    config = config.with_overrides(**overrides)

    # define the model and collectors path for saving after training
    model_path = models_dir / f"VICReg_model_{config.epochs}_epochs_{suffix}.pth"
    collectors_path = collectors_dir / f"VICReg_model_collectors_{config.epochs}_epochs_{suffix}.pkl"

    # train the model if specified.
    if train:
        print(f"Path to save the trained model: {model_path}")
        print(f"Path to save the collectors after model training: {collectors_path}")
        # get train and test loaders
        trainset = data_setup.CustomCIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
        train_loader = DataLoader(trainset, batch_size=config.batch_size, shuffle=True, num_workers=2)
        testset = data_setup.CustomCIFAR10(root=data_dir, train=False, download=True, transform=train_transform)
        test_loader = DataLoader(testset, batch_size=config.batch_size, shuffle=False, num_workers=2)
        # initialize the VICRegModel with projector dimension and encoder dimension.
        model = VICRegModel(D=config.D, proj_dim=config.proj_dim).to(device)
        # initialize criterion instance for VICReg with the given hyper-parameters
        criterion = VICRegCriterion(lam=config.lambda_,
                                    mu=config.mu,
                                    nu=config.nu,
                                    gamma=config.gamma,
                                    eps=config.eps)
        # initialize the optimizer with the model parameters
        optimizer = config.get_optimizer(model.parameters())

        # train the
        vicreg_engine.train(model,
              train_loader=train_loader,
              test_loader=test_loader,
              criterion=criterion,
              optimizer=optimizer,
              epochs=config.epochs,
              device=device,
              save=True,
              model_path = model_path,
              collectors_path =collectors_path
              )

    # load trained model and collectors if the model exists.
    if model_path.exists():
        # model = VICRegModel(D=config.D, proj_dim=config.proj_dim).to(device)
        # model = utils.load_model(model, model_path=model_path, device=device)
        collectors = utils.load_collectors(collectors_path=collectors_path)
        # plot losses
        if show_plot:
            plot_losses(config.epochs, pd.DataFrame(collectors), plot_path)
        return model_path, collectors_path
    raise BaseException("Model path doesn't exist.. train the model first")

def q2_pca_vs_t_sne_visualizations(config, vicreg_model_path, plot_path, show_plot=True):
    # load trained VICReg Model
    model = VICRegModel(D=config.D, proj_dim=config.proj_dim).to(device)
    model = utils.load_model(model, model_path=vicreg_model_path, device=device)

    # get testset and test loader (as single batch of all test points)
    orig_testset = data_setup.OriginalCIFAR10(root=data_dir, train=False, transform=test_transform,download=True)
    test_loader = torch.utils.data.DataLoader(orig_testset, batch_size=len(orig_testset), shuffle=False, num_workers=2)
    images, labels = next(iter(test_loader))

    # get images representations
    model.eval()
    with torch.inference_mode():
        Y = model.encoder.encode(images.to(device)).cpu().numpy()

    # compute dim reductions for image representations into 2D space
    Y_reduced_pca = perform_pca_or_tsne(Y=Y, n_components=2, dim_reduction_type="pca")
    Y_reduced_tsne = perform_pca_or_tsne(Y=Y, n_components=2, dim_reduction_type="tsne")

    # show plot if specified
    if show_plot:
        plot_dimensionality_reduction(Y_reduced_pca, Y_reduced_tsne, labels, plot_path)

def q3_linear_probing(config,overrides=None, train_classifier=True, vicreg_model_path=Path(""),suffix=""):
  # override config hyper-parameters if specified for training the classifier
  classifier_config = config.with_overrides(**overrides) if overrides else config

  # define the classifier model path
  classifier_model_path = models_dir / f"VICReg_linear_probing_{classifier_config.epochs}_epochs_{suffix}.pth"

  # load original VICRegModel
  vicreg_model = VICRegModel(D=classifier_config.D, proj_dim=classifier_config.proj_dim).to(device)
  vicreg_model = utils.load_model(vicreg_model,model_path=vicreg_model_path, device=device)

  # initialize VICReg classifier model based on the original model.
  classifier_model = VICRegLinearProbing(vicreg_model, D= classifier_config.D, num_classes=len(CIFAR10_CLASSES)).to(device)
  if train_classifier:
    # get train loader from the original trainset
    orig_trainset = data_setup.OriginalCIFAR10(root=data_dir, train=True, transform=train_transform)
    train_loader = DataLoader(orig_trainset, batch_size=classifier_config.batch_size, shuffle=True, num_workers=2)

    # freeze the the parameters of the model encoder
    utils.freeze_encoder_only(classifier_model.encoder)

    #initialize optimizer and criterion instances
    optimizer = classifier_config.get_optimizer(classifier_model.parameters())
    criterion = torch.nn.CrossEntropyLoss().to(device)
    epochs = classifier_config.epochs
    print(f"Path to save the trained classifier: {classifier_model_path}")

    # train the classifier via linear probing for specified number of epoches
    classifier_model = vicreg_engine.train_classifier(classifier_model,
                                              train_loader,
                                              criterion,
                                              optimizer,
                                              epochs,
                                              device,
                                              model_path=classifier_model_path)
  else:
    # Load the trained classifier model
    classifier_model = utils.load_model(classifier_model, classifier_model_path, device=device,verbose=True)

  # evaluating the probing's accuracy on the test set
  orig_testset = data_setup.OriginalCIFAR10(root=data_dir, train=False, transform=test_transform)
  test_loader = DataLoader(orig_testset, batch_size=len(orig_testset), shuffle=False, num_workers=2)
  accuracy = vicreg_engine.test_classifier(classifier_model, test_loader, device)
  print(f"The model after linear probing achieved {np.round(accuracy, 2)*100}% accuracy on the test set of CIFAR10")
  print(f"Trained classifer saved in: {models_dir / classifier_model_path}")
  return classifier_model_path

def q4_ablation_1_no_variance_term(config, overrides=None, train_vicreg=False, train_classifier=False):
    # override config hyper-parameters if specified
    config = config.with_overrides(**overrides) if overrides else config

    # train VICReg model from scratch with no variance (mu=0)
    vicreg_model_path, _ = q1_training(config,
                                       overrides={"epochs": 30, "mu": 0},
                                       train=train_vicreg,
                                       plot_path=plots_dir / "q4_training_losses.png",
                                       suffix="no_var",
                                       show_plot=False
                                       )

    q2_pca_vs_t_sne_visualizations(config,
                                   vicreg_model_path,
                                   plot_path=plots_dir / "q4_pca_vs_tsne_visualizations_no_var.png",
                                   show_plot=True)

    classifier_path = q3_linear_probing(config,
                                        overrides={"epochs": 3},
                                        train_classifier=train_classifier,
                                        vicreg_model_path=vicreg_model_path,
                                        suffix="no_var")
    return vicreg_model_path, classifier_path

def q5_ablation_2_no_generated_neighbors(config, vicreg_model_path, train_vicreg=False, train_classifier=False,k=4):
    # Load original trained VICReg model
    vicreg_model = VICRegModel(D=config.D, proj_dim=config.proj_dim).to(device)
    vicreg_model = utils.load_model(vicreg_model, model_path=vicreg_model_path, device=device)
    repr_path = data_dir / f"train_CIFAR10_Y_by_vicreg_{config.epochs}"
    indices_path = data_dir / f"train_CIFAR10_{k - 1}_NN_indices_vicreg_{config.epochs}"
    # Compute the representations of this VICReg model, on all the training set
    Y = get_image_representations(vicreg_model, filename=repr_path)  # Y.shape: (50000, 128)
    indices = get_nearest_neighbors(Y, filename=indices_path, k=k)  # indices.shape: (50000, 4)
    # Moving the original VICReg model to cpu to save both run time and GPU space.
    vicreg_model.to(device='cpu')

    # define the model and collectors path for saving after training
    config = config.with_overrides(**{"epochs": 1})
    model_ngn_path = models_dir / f"VICReg_model_{config.epochs}_epochs_no_gen_neighbors.pth"
    collectors_ngn_path = collectors_dir / f"VICReg_model_collectors_{config.epochs}_epochs_no_gen_neighbors.pkl"
    if train_vicreg:
        # get modified version of CIFAR10 which stores the neighbors indices.
        # each item represented as: (image, nn_image, label)
        # where nn_image is randomly selected nearest neighbor from indices.
        orig_trainset= data_setup.OriginalCIFAR10(root=data_dir, train=True, transform=test_transform, download=True)
        trainset = data_setup.CIFAR10WithNeighbors(dataset=orig_trainset, neighbors_indices=indices)
        train_loader = DataLoader(trainset, batch_size=config.batch_size, shuffle=True, num_workers=2)

        # initialize criterion instance for VICReg with the given hyper-parameters.
        criterion = VICRegCriterion(lam=config.lambda_,
                                    mu=config.mu,
                                    nu=config.nu,
                                    gamma=config.gamma,
                                    eps=config.eps)

        # initialize the VICRegModel with projector dimension and encoder dimension.
        model = VICRegModel(D=config.D, proj_dim=config.proj_dim).to(device)

        # initialize the optimizer with the model parameters.
        optimizer = config.get_optimizer(model.parameters())
        # train the
        vicreg_engine.train(model,
                            train_loader=train_loader,
                            test_loader=train_loader,
                            criterion=criterion,
                            optimizer=optimizer,
                            epochs=config.epochs,
                            device=device,
                            save=True,
                            model_path=model_ngn_path,
                            collectors_path=collectors_ngn_path
                            )

    # load trained model and collectors if the model exists.
    if model_ngn_path.exists():
        model = VICRegModel(D=config.D, proj_dim=config.proj_dim).to(device)
        model = utils.load_model(model, model_path=model_ngn_path, device=device)
        print(f"{model_ngn_path} loaded!")

    classifier_path = q3_linear_probing(config,
                                        overrides={"epochs": 3},
                                        train_classifier=train_classifier,
                                        vicreg_model_path=model_ngn_path,
                                        suffix="no_gen")

    return model_ngn_path, classifier_path

def q7_retrieval_evaluation(config, models_paths):
    # Load original CIFAR10 trainset and trainloader
    trainset = data_setup.OriginalCIFAR10(root=data_dir, train=True)
    train_loader = DataLoader(trainset, batch_size=config.batch_size, shuffle=False, num_workers=2)

    # Select 10 random images from the training set, one from each class
    random_images, random_labels, random_ind = get_random_images(train_loader)

    k = 6  # set k for KNN
    # For each selected image, use the representations of each of the evaluated methods
    # [VICReg, No Generated Neighbors], to find its 5 nearest neighbors in the trainset
    # Load models
    models = []
    for i in range(2):
        print(models_paths[i])
        model = VICRegModel(D=config.D, proj_dim=config.proj_dim).to(device)
        model = utils.load_model(model, model_path=models_paths[i], device=device)
        models.append(model)

    titles = ["Original VICReg Model", "VICReg with no Generated Neighbors"]
    repr_paths = [data_dir / f"train_CIFAR10_Y_by_vicreg_{config.epochs}",
                  data_dir / "train_CIFAR10_Y_by_vicreg_ngn_1_epoch"]

    indices_paths = [data_dir / f"train_CIFAR10_{k - 1}_NN_indices_vicreg_{config.epochs}",
                     data_dir / f"train_CIFAR10_{k - 1}_NN_indices_vicreg_ngn_1_epoch"]
    plot_paths = [plots_dir / "q7_retrieval_evaluation_orig.png", plots_dir / "q7_retrieval_evaluation_ngn.png"]
    for i in range(2):
        Y, indices = get_representations_and_nn_indices(models[i], config.batch_size, repr_paths[i], indices_paths[i],
                                                        k=k)
        queries = Y[random_ind]
        nearest_neighbors = find_neighbors(Y, queries, k=k)  # list of 10 arrays each 6 elements
        farthest_neighbors = find_neighbors(Y, queries, k=k, return_farthest=True)  # list of 10 arrays each 5 elements
        plot_image_nn(trainset, random_images, random_labels, nearest_neighbors, farthest_neighbors, k=k,
                      title=titles[i], plot_path=plot_paths[i])


def main():
    # get base config with default hyper-parameters
    base_config = Config()
    print("Q1: Training.")
    vicreg_model_path, _ = q1_training(config=base_config,
                                       overrides={"epochs": 30},
                                       train=False,
                                       plot_path=plots_dir / "q1_training.png",
                                       show_plot=True,
                                      )
    print("Q2: PCA vs T-SNE Visualizations.")
    q2_pca_vs_t_sne_visualizations(config=base_config,
                                   vicreg_model_path=vicreg_model_path,
                                   plot_path=plots_dir / "q2_pca_vs_tsne_visualizations.png",
                                   show_plot=True)
    print("Q3: Linear Probing.")
    q3_linear_probing(config=base_config,
                     overrides={"epochs": 3},
                     train_classifier=False,
                     vicreg_model_path=vicreg_model_path)
    # The model after linear probing achieved 70.0% accuracy on the test set of CIFAR10
    # Trained classifer saved in: models/models/VICReg_linear_probing_3_epochs_.pth
    print("Q4: Ablation 1 - No Variance Term.")
    q4_ablation_1_no_variance_term(config=base_config,
                                  train_vicreg=False,
                                  train_classifier=False)

    # # The model after linear probing achieved 14.000000000000002% accuracy on the test set of CIFAR10
    # # trained classifer saved at path: /VICReg_linear_probing_3_epochs_no_var.pth
    print("Q5: Ablation 2 - No Generated Neighbors.")

    vicreg_model_path_ngn, _ = q5_ablation_2_no_generated_neighbors(config=base_config,
                                                                    vicreg_model_path=vicreg_model_path,
                                                                    train_vicreg=False,
                                                                    train_classifier=False)
    print("Q7: Retrieval Evaluation.")
    models_paths = [vicreg_model_path,vicreg_model_path_ngn]
    q7_retrieval_evaluation(base_config,models_paths=models_paths)
    print("Done.")
if __name__ == '__main__':
    main()