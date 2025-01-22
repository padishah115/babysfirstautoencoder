import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from autoencoder import Autoencoder
from data import load_MNIST
import matplotlib.pyplot as plt

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

    return mean_val_loss.item()


def load_and_validate(central_chans:int):
    """
    
    Args:
        central_chans (int): The number of central channels on the model we want to load
    """

    model_path=f"./trained_models/autoencoder_{central_chans}_central_chans.pt"

    #First get the training loss.
    loaded_model = Autoencoder(central_chans=central_chans)
    loaded_model.load_state_dict(torch.load(model_path, weights_only=False))

    #Get data about the number of parameters in the model.
    # numel_list = [p.numel() for p in loaded_model.parameters()]
    # print(numel_list)

    _, val_loader = load_MNIST()

    mean_val_loss = validate(
        model=loaded_model,
        val_loader=val_loader
        )
    
    print(f"Mean validation loss, {central_chans} Central Channels: {mean_val_loss:.4f}")



def plot_sample_outputs(central_chans:int, no_of_images:int=5):
    """
    Plot some sample outputs from the data.

    Args:
        chan_no (int): The channel number for the model we are testing.
        no_of_images (int): The number of images from the validation set that we want to generate mimicries of.
    """

    model_path = f'./trained_models/autoencoder_{central_chans}_central_chans.pt'

    loaded_model = Autoencoder(central_chans=central_chans)
    loaded_model.load_state_dict(torch.load(model_path, weights_only=False))

    _, val_loader = load_MNIST()

    imgs_and_labels = [(batch, labels) for batch, labels in val_loader]

    imgs = []

    for i in range(no_of_images):
        imgs.append(imgs_and_labels[0][0][i][0])

    inputs = [img.view(-1, 784) for img in imgs]
    outputs = [loaded_model(input) for input in inputs]

    #One row for each image in the validation set.
    fig, axs = plt.subplots(nrows=no_of_images, ncols=2)

    axs[0][0].set_title('Input')
    axs[0][1].set_title('Model output')

    for i, img in enumerate(imgs):
        axs[i][0].imshow(img.detach().numpy())
        axs[i][0].axis("off")

        axs[i][1].imshow(outputs[i].view(28,28).detach().numpy())
        axs[i][1].axis("off")

    fig.suptitle(f'Performance on MNIST Dataset: {central_chans} Central Channels')
    fig.tight_layout()
    plt.savefig(f'./plots/{central_chans}_channels.png')

            

        