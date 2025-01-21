##############################################################
# Load a pre-trained model and then observe its performance. #
##############################################################

import torch
from autoencoder import Autoencoder
from data import load_MNIST
from train_and_validate.validate import validate

def main(model_path:str="./trained_models/autoencoder.pt"):

    loaded_model = Autoencoder()
    loaded_model.load_state_dict(torch.load(model_path))

    #Get data about the number of parameters in the model.
    # numel_list = [p.numel() for p in loaded_model.parameters()]
    # print(numel_list)

    _, val_loader = load_MNIST()

    mean_val_loss = validate(
        model=loaded_model,
        val_loader=val_loader
        )
    
    print(mean_val_loss.detach())


if __name__ == "__main__":
    main()