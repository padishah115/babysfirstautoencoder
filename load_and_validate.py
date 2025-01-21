##############################################################
# Load a pre-trained model and then observe its performance. #
##############################################################

import torch
from autoencoder import Autoencoder
from test_and_train.test import test

def main(model_path:str="./trained_models/autoencoder.pt"):

    loaded_model = Autoencoder()
    loaded_model.load_state_dict(torch.load(model_path))

    numel_list = [p.numel() for p in loaded_model.parameters()]
    print(numel_list)


if __name__ == "__main__":
    main()