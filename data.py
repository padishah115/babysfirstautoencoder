import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from typing import Tuple

#data_path = './data'

def load_MNIST(data_path:str='./data', batch_size:int=256)->Tuple[DataLoader, DataLoader]:
    """
    Function which loads the handwritten digit images from the MNIST database.

    Args:
        data_path (str): The path at which the MNIST data will be downloaded.

    Returns:
        train_loader (DataLoader): The training set wrapped in a DataLoader.
        val_loader (DataLoader): The validation set wrapped in a DataLoader.

    """

 
    transform = transforms.ToTensor()
    
    #Load the mnist_unnorm dataset
    mnist = datasets.MNIST(
        root=data_path,
        train=True,
        download=True,
        transform=transform
    )


    mnist_val = datasets.MNIST(
        root=data_path,
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(mnist, batch_size=batch_size, shuffle=True) #Shuffle the training set- increase diversity of training set
    val_loader = DataLoader(mnist_val, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader




