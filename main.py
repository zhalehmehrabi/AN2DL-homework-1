import os

import optuna
from optuna.trial import TrialState
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import models
from torchvision import transforms, datasets

import matplotlib.pyplot as plt

DEVICE = torch.device("cuda")
BATCHSIZE = 128
CLASSES = 10
DIR = os.getcwd()
EPOCHS = 10
N_TRAIN_EXAMPLES = BATCHSIZE * 30
N_VALID_EXAMPLES = BATCHSIZE * 10


def define_model(trial):
    # We optimize the number of layers, hidden units and dropout ratio in each layer.
    n_layers = trial.suggest_int("n_layers", 1, 3)
    layers = []

    in_features = 28 * 28
    for i in range(n_layers):
        out_features = trial.suggest_int("n_units_l{}".format(i), 4, 128)
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        p = trial.suggest_float("dropout_l{}".format(i), 0.2, 0.5)
        layers.append(nn.Dropout(p))

        in_features = out_features
    layers.append(nn.Linear(in_features, CLASSES))
    layers.append(nn.LogSoftmax(dim=1))

    return nn.Sequential(*layers)


def get_dataset(train_batch):

    data_dir = "./training_data_final"

    # perform some transformations like resizing,
    # centring and tensorconversion
    # using transforms function
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = transforms.Compose(
        [transforms.Resize(255),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize(mean, std)])


    # pass the image data folder and
    # transform function to the datasets
    # .imagefolder function
    dataset = datasets.ImageFolder(data_dir,
                                   transform=transform)
    train_set, valid_set, test_set = torch.utils.data.random_split(dataset, [0.7, 0.2, 0.1], generator=torch.Generator().manual_seed(42))

    # now use dataloder function load the
    # dataset in the specified transformation.
    train_loader = torch.utils.data.DataLoader(train_set,
                                             batch_size=train_batch,
                                             shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_set,
                                               batch_size=32,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set,
                                               batch_size=32,
                                               shuffle=True)

    # specify the image dataset folder
    return train_loader, valid_loader, test_loader
    # return train_loader, valid_loader


def objective(trial):

    # Generate the model.
    model = define_model(trial).to(DEVICE)

    # Generate the optimizers.
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    # Get the FashionMNIST dataset.
    train_loader, valid_loader = get_dataset(train_batch=32)

    # Training of the model.
    for epoch in range(EPOCHS):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            # Limiting training data for faster epochs.
            if batch_idx * BATCHSIZE >= N_TRAIN_EXAMPLES:
                break

            data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)

            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

        # Validation of the model.
        model.eval()
        correct = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(valid_loader):
                # Limiting validation data.
                if batch_idx * BATCHSIZE >= N_VALID_EXAMPLES:
                    break
                data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)
                output = model(data)
                # Get the index of the max log-probability.
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        accuracy = correct / min(len(valid_loader.dataset), N_VALID_EXAMPLES)

        trial.report(accuracy, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return accuracy


if __name__ == "__main__":
    # study = optuna.create_study(direction="maximize")
    # study.optimize(objective, n_trials=100, timeout=600)
    #
    # pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    # complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    #
    # print("Study statistics: ")
    # print("  Number of finished trials: ", len(study.trials))
    # print("  Number of pruned trials: ", len(pruned_trials))
    # print("  Number of complete trials: ", len(complete_trials))
    #
    # print("Best trial:")
    # trial = study.best_trial
    #
    # print("  Value: ", trial.value)
    #
    # print("  Params: ")
    # for key, value in trial.params.items():
    #     print("    {}: {}".format(key, value))
    train_loader, valid_loader, test_loader = get_dataset(train_batch=32)
    model = models.resnet18(pretrained=True)
    # print(model)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(512, 8)
    optimizer = torch.optim.Adam([parameters for parameters in model.parameters() if parameters.requires_grad], lr=0.003)
    N_EPOCHS = 20
    cost_list = []
    accuracy_list = []
    correct = 0
    n_test = len(valid_loader)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(N_EPOCHS):
        COST = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            z = model(x)
            loss = criterion(z, y)
            loss.backward()
            optimizer.step()
            COST += loss.data

        cost_list.append(COST)
        correct = 0
        # perform a prediction on the validation  data
        for x_test, y_test in valid_loader:
            z = model(x_test)
            _, yhat = torch.max(z.data, 1)
            correct += (yhat == y_test).sum().item()
        accuracy = correct / n_test
        accuracy_list.append(accuracy)
        print(epoch)
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.plot(cost_list, color=color)
    ax1.set_xlabel('epoch', color=color)
    ax1.set_ylabel('Cost', color=color)
    ax1.tick_params(axis='y', color=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('accuracy', color=color)
    ax2.set_xlabel('epoch', color=color)
    ax2.plot(accuracy_list, color=color)
    ax2.tick_params(axis='y', color=color)
    fig.tight_layout()
