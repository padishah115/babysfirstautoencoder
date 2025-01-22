#############################################################################
# Program for loading the MNIST database and then training the Autoencoder. #
#############################################################################

import torch
import torch.optim as optim
from data import load_MNIST
from autoencoder import Autoencoder
from train_and_validate.train import train
from train_and_validate.load_and_validate import load_and_validate


#Prevent access errors when trying to load the MNIST database
import ssl 
ssl._create_default_https_context = ssl._create_unverified_context


def main():

    data_path = './data'
    train_loader, val_loader = load_MNIST(data_path=data_path, batch_size=256) 

    #Load autoencoder model
    myAutoencoder = Autoencoder()
    n_epochs = 30
    learning_rate = 1e-3
    optimizer = optim.Adam(params=myAutoencoder.parameters(), lr=learning_rate)

    #Training loop
    train(
        n_epochs=n_epochs,
        model=myAutoencoder,
        optimizer=optimizer,
        train_loader=train_loader
    )

    #Save the model
    save_path = './trained_models/'
    torch.save(myAutoencoder.state_dict(), save_path + 'autoencoder.pt')

    #Load the model and validate
    load_and_validate()


if __name__ == '__main__':
    main()

