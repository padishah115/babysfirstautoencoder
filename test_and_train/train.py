import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

def train(n_epochs:int, model:nn.Module, optimizer:optim.Optimizer, train_loader:DataLoader):
    """
    Performs training loops over specified number of epochs using desired optimizer.

    Args:
        n_epochs (int): Number of training epochs
        model (nn.Module): The model to be trained (in this project, the autoencoder).
        optimizer (optim.Optimizer): The desired optimization algorithm for backpropagation.
        train_loader (DataLoader): The training set wrapped in a DataLoader.

    """

    for epoch in range(1, n_epochs+1):
        
        loss_train = 0.0
        
        for imgs, labels in train_loader:

            batch_size = imgs.shape[0]
            img_batch = imgs.view(batch_size, -1)
            
            outputs = model(img_batch)
            loss = model.loss_fn(outputs, img_batch) #calculate the BCE Loss

            optimizer.zero_grad() #prevent gradient accumulation
            loss.backward() #backpropagation on the model
            optimizer.step()
            
            loss_train += loss

        if epoch == 1 or epoch % 10 == 0:
            print(f"Mean batch loss at epoch {epoch}", loss_train/len(train_loader)) #print the batch-normalised loss at epoch 1 and every 10th epoch

    