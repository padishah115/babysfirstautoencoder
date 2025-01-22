#############################################################################
# Program for loading the MNIST database and then training the Autoencoder. #
#############################################################################

import torch
import torch.optim as optim
from data import load_MNIST
from autoencoder import Autoencoder
from train_and_validate.train import train
from train_and_validate.validate import load_and_validate, plot_sample_outputs


#Prevent access errors when trying to load the MNIST database
import ssl 
ssl._create_default_https_context = ssl._create_unverified_context


def main():

    #The different quantities of central channels that we want to use
    chans = [16, 32, 48, 64]

    data_path = './data'
    train_loader, val_loader = load_MNIST(data_path=data_path, batch_size=256) 
    

    n_epochs = 30
    learning_rate = 1e-3

    for central_chans in chans:

        #Load autoencoder model
        myAutoencoder = Autoencoder(central_chans=central_chans)

        #Use adam optimizer
        optimizer = optim.Adam(params=myAutoencoder.parameters(), lr=learning_rate)

        #Training loop
        print(f'\nTraining model with {central_chans} central channels:')
        train(
            n_epochs=n_epochs,
            model=myAutoencoder,
            optimizer=optimizer,
            train_loader=train_loader
        )

        #Save the model
        save_path = './trained_models'
        torch.save(myAutoencoder.state_dict(), save_path + f'/autoencoder_{central_chans}_central_chans.pt')

        #Load the model and validate
        load_and_validate(central_chans=central_chans)

    for central_chans in chans:
        plot_sample_outputs(central_chans=central_chans)


if __name__ == '__main__':
    main()

