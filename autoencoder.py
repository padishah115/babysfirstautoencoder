import torch
import torch.nn as nn

class myAutoencoder(nn.Module):
    """
    Autoencoder class for processing images from the MNIST database. Requires flattened 1D image vector.

    Attributes:
        outer_chans (int): The dimensions of the input/output FC linear layers. Default is 784 (MNIST images are 28x28)
        central_chans (int): The dimensions of the central layer.

    """

    def __init__(self, outer_chans=784, central_chans=16):
        super().__init__()
        self.outer_chans = outer_chans
        self.central_chans = central_chans

        self.input_layer = nn.Linear(in_features=self.outer_chans, out_features=self.central_chans)
        self.output_layer = nn.Linear(in_features=self.central_chans, out_features=self.outer_chans)
        self.activation = nn.Sigmoid()

        #use binary cross entropy loss function for autoencoder training.
        self.loss_fn = nn.BCELoss()


    def forward(self, x):
        out = self.activation(self.output_layer(self.input_layer(x)))
        return out
    
