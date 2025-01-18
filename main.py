import torch
import torch.optim as optim
from data import load_MNIST
from autoencoder import myAutoencoder
from train import train


def main():

    data_path = './data'
    train_loader, val_loader = load_MNIST(data_path=data_path, normalise=True) #pixels normalised to between 0 and 1

    #Load autoencoder model
    model = myAutoencoder()
    n_epochs = 100
    learning_rate = 1e-2
    optimizer = optim.SGD(params=model.parameters(), lr=learning_rate)

    train(
        n_epochs=n_epochs,
        model=model,
        optimizer=optimizer,
        train_loader=train_loader
    )



if __name__ == '__main__':
    main()

