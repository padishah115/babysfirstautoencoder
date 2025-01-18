import torch
import torch.nn as nn
from torch.utils.data import DataLoader

#Need to think a little more about what I should be doing with this.

# def test(model:nn.Module, val_loader:DataLoader, generation_no:int)->torch.Tensor:
#     """
#     Tests the output of the autoencoder.

#     Args:
#         model (nn.Module): The autoencoder model to be trained.
#         val_image (torch.Tensor): The validation images we would like to test the model against.
#         generation_no (int): The number of images we want the model to generate.

#     Returns:
#         output (torch.Tensor): The generated output in tensor form.

#     """

#     output_list = []

#     with torch.no_grad():
#         for val_images, _ in val_loader:
#             no_per_batch = int(generation_no/len(val_images))

#             for img in val_images[:no_per_batch]:
#                 output = model(img)


#         # image = image.view(-1, 784)
#         # output = model(image)
#         # output_list.append(output)

#     output_stack = torch.cat(output_list, dim=0)

#     return output_stack