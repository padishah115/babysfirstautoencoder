import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from autoencoder import Autoencoder

def train(n_epochs:int, model:Autoencoder, optimizer:optim.Optimizer, train_loader:DataLoader):
    """
    Performs training loops over specified number of epochs using desired optimizer.

    Args:
        n_epochs (int): Number of training epochs
        model (nn.Module): The model to be trained (in this project, the autoencoder).
        optimizer (optim.Optimizer): The desired optimization algorithm for backpropagation.
        train_loader (DataLoader): The training set wrapped in a DataLoader.

    """

    criterion = model.criterion #Loss function for the model

    for epoch in range(1, n_epochs+1):
        
        loss_train = 0.0
        
        for imgs, labels in train_loader:

            batch_size = imgs.shape[0]
            inputs = imgs.view(batch_size, -1)
            
            outputs = model(inputs)
            loss = criterion(outputs, inputs) #calculate the BCE Loss

            optimizer.zero_grad() #prevent gradient accumulation
            loss.backward() #backpropagation on the model
            optimizer.step()
            
            loss_train += loss.item()

        if epoch == 1 or epoch % 10 == 0:
            print(f"Mean batch loss at epoch {epoch}", loss_train/len(train_loader)) #print the batch-normalised loss at epoch 1 and every 10th epoch

    