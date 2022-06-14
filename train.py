from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.losses import DiceLoss, DiceCELoss


import torch
from preporcess import prepare
from utilities import train,calculate_weights


data_dir = 'C:/Users/mihne/Desktop/Accenture-Collab/Datasets/COVID-19-20_v2/Diicom'
model_dir = 'C:/Users/mihne/Desktop/Accenture-Collab/LiverSeg-using-monai-and-pytorch/Liver-Segmentation-Using-Monai-and-PyTorch/results' 
data_in = prepare(data_dir, cache=True)

device = torch.device("cuda")
model = UNet(
    dimensions=3, #we are working with 3D Images (Volumes)
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128, 256), 
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH,
).to(device)


#loss_function = DiceCELoss(to_onehot_y=True, sigmoid=True, squared_pred=True, ce_weight=calculate_weights(1792651250,2510860).to(device))
loss_function = DiceLoss(to_onehot_y=True, sigmoid=True, squared_pred=True)
optimizer = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=1e-3, amsgrad=True)

if __name__ == '__main__':
    train(model, data_in, loss_function, optimizer, 300, model_dir)

#improvements:
#dividing each patient into groups( each patient has a different muber of slices,diicom files) eg: make groups of 60 slices - done
#change the optimizer
#change the loss_function
#change the learning rate -done
#best is to add more data
#change the contrast when preprocessing data
#data augmentation - done
#use compressed nii files (nii,gz) - done

