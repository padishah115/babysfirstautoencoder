import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from typing import Tuple


#data_path = './data'

def load_MNIST(data_path:str='./data', normalise:bool=True)->Tuple[DataLoader, DataLoader]:
    """
    Function which loads the handwritten digit images from the MNIST database.

    Args:
        data_path (str): The path at which the MNIST data will be downloaded.
        normalise (bool): Determines whether the image pixels will be normalised to values between 0 and 1.

    Returns:
        train_loader (DataLoader): The training set wrapped in a DataLoader.
        val_loader (DataLoader): The validation set wrapped in a DataLoader.

    """
    
    #Load the mnist_unnorm dataset
    mnist_unnorm = datasets.MNIST(
        root=data_path,
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )

    if normalise:
        #Divides all pixels by 0, 1 so that we can use the softmax or sigmoid activation layers on the output
        mnist_norm = datasets.MNIST(
            root=data_path,
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x/255) #Divide by 255 (max pixel value)
            ])
        )

        mnist_val_norm = datasets.MNIST(
            root=data_path,
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x/255) #Divide by 255 (max pixel value)
            ])
        )

        train_loader = DataLoader(mnist_norm)
        val_loader = DataLoader(mnist_val_norm)

        return train_loader, val_loader

    else:
        #If the user doesn't want normalization.
        mnist_val_unnorm = datasets.MNIST(
            root=data_path,
            train=False,
            download=True,
            transform=transforms.ToTensor()
        )

        train_loader = DataLoader(mnist_unnorm, batch_size=64, shuffle=False)
        val_loader = DataLoader(mnist_val_unnorm, batch_size=64, shuffle=False)

        return train_loader, val_loader



