import torch
import torch.nn as nn
import torch.nn.functional as F

class Autoencoder(nn.Module):
    """
    Autoencoder class for processing images from the MNIST database. Requires flattened 1D image vector.

    Attributes:
        outer_chans (int): The dimensions of the input/output FC linear layers. Default is 784 (MNIST images are 28x28)
        central_chans (int): The dimensions of the central layer.

    """

    def __init__(self, outer_chans=784, central_chans=16):
        super().__init__()
        self.outer_chans = outer_chans #Number of pixels in flattened image tensor
        self.central_chans = central_chans #Number of channels in the central layer

        self.encoder = nn.Linear(in_features=self.outer_chans, out_features=self.central_chans)
        self.decoder = nn.Linear(in_features=self.central_chans, out_features=self.outer_chans)

        #use binary cross entropy loss function for autoencoder training.
        self.criterion = nn.MSELoss()


    def forward(self, x:torch.Tensor)->torch.Tensor:
        x = F.relu(self.encoder(x)) #Use the relu activation function for the output from the encoder
        x = F.sigmoid(self.decoder(x)) #Use the sigmoid activation function on the final images as they are normalized to between 0, 1
        return x
    
