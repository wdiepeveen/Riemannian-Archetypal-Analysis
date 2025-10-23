import os
import torch
from nflows import flows, transforms, distributions
from nflows.transforms.normalization import ActNorm

from src.diffeomorphisms.image.nflow import NFlowImageDiffeomorphism
from src.transforms.multi_layer_parity_conv2d.tanh import MultiLayerTanhParityConv2DTransform
from src.transforms.parity_conv import ParityConv2DTransform

class MNIST28NFlowImageDiffeomorphism(NFlowImageDiffeomorphism):
    def __init__(self):

        model = self.create_flow_model(1, 28, 28, 5, 128, 6, 6)
        phi = NFlowImageDiffeomorphism(1, 28, 28, model)

        model_path = os.path.join(os.path.join('models','mnist_28x28','model.pth'))
        checkpoint = torch.load(model_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
        phi.load_state_dict(checkpoint['model_state_dict'])
        phi.eval()

        super().__init__(1, 28, 28, phi.nflow)


    def create_flow_model(self,in_channels, height, width, kernel_size, latent_channels, order, n_flows, unit_det=False, parity_equivariance=False, actnorm=True):
        base_dist = distributions.StandardNormal(shape=[in_channels, height, width])
        transforms_list = []
        for i in range(n_flows):
            if actnorm:
                transforms_list.append(ActNorm(features=in_channels))
            transforms_list.append(ParityConv2DTransform(in_channels, height, width, kernel_size, parity=i, unit_det=unit_det, parity_equivariance=parity_equivariance))
            transforms_list.append(ParityConv2DTransform(in_channels, height, width, kernel_size, parity=i+1, unit_det=unit_det, parity_equivariance=parity_equivariance))
            transforms_list.append(MultiLayerTanhParityConv2DTransform(in_channels, height, width, kernel_size, latent_channels, order=order, parity=i))
        if actnorm:
                transforms_list.append(ActNorm(features=in_channels))
        transforms_list.append(ParityConv2DTransform(in_channels, height, width, kernel_size, parity=n_flows, unit_det=unit_det, parity_equivariance=parity_equivariance))
        transforms_list.append(ParityConv2DTransform(in_channels, height, width, kernel_size, parity=n_flows+1, unit_det=unit_det, parity_equivariance=parity_equivariance))
        flow_transforms = transforms.CompositeTransform(transforms_list)
        return flows.Flow(transform=flow_transforms, distribution=base_dist)