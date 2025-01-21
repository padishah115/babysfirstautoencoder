#############################################################################
# Program for loading the MNIST database and then training the Autoencoder. #
#############################################################################

import torch
import torch.optim as optim
from data import load_MNIST
from autoencoder import Autoencoder
from test_and_train.train import train


#Prevent access errors when trying to load the MNIST database
import ssl 
ssl._create_default_https_context = ssl._create_unverified_context


def main():

    data_path = './data'
    train_loader, val_loader = load_MNIST(data_path=data_path, normalise=True) #pixels normalised to between 0 and 1

    #Load autoencoder model
    myAutoencoder = Autoencoder()
    n_epochs = 100
    learning_rate = 1e-2
    optimizer = optim.SGD(params=myAutoencoder.parameters(), lr=learning_rate)

    train(
        n_epochs=n_epochs,
        model=myAutoencoder,
        optimizer=optimizer,
        train_loader=train_loader
    )

    save_path = './trained_models/'
    torch.save(myAutoencoder.state_dict(), save_path + 'autoencoder.pt')


if __name__ == '__main__':
    main()

