import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from autoencoder import Autoencoder

from data import load_MNIST

#Need to think a little more about what I should be doing with this.

def validate(model:Autoencoder, val_loader:DataLoader)->float:
    """
    Tests the output of the autoencoder.

    Args:
        model (nn.Module): The autoencoder model to be trained.
        val_image (torch.Tensor): The validation images we would like to test the model against.
        generation_no (int): The number of images we want the model to generate.

    Returns:
        mean_val_loss: The total loss across the validation set, divided by the number of images in the validation set.

    """

    loss_val = 0.0
    total = 0

    with torch.no_grad():
        for val_images, _ in val_loader:

            batch_size = val_images.shape[0] #get the size of the batch.
            total += batch_size

            inputs = val_images.view(batch_size, -1)
            
            outputs = model(inputs)
            loss_val += model.criterion(outputs, inputs)

    mean_val_loss = loss_val / total

    return mean_val_loss

            

        