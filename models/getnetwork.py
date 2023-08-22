import sys
from models import *
import torch.nn as nn

def get_network(network, in_channels, num_classes, **kwargs):

    # 2d networks
    if network == 'unet':
        net = unet(in_channels, num_classes)
    # 3d networks
    elif network == 'unet3d':
        net = unet3d(in_channels, num_classes)
    else:
        print('the network you have entered is not supported yet')
        sys.exit()
    return net
