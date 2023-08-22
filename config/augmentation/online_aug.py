import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchio import transforms as T
import torchio as tio

def data_transform_3d(normalization):
    data_transform = {
        'train': T.Compose([
            T.RandomFlip((0, 1)),
            T.RandomBiasField(coefficients=(0.12, 0.15), order=2, p=0.2),
            T.OneOf({
               T.RandomNoise(): 0.5,
               T.RandomBlur(std=1): 0.5,
            }, p=0.2),
            T.ZNormalization(masking_method=normalization),
        ]),
        'val': T.Compose([
            T.ZNormalization(masking_method=normalization),
        ]),
        'test': T.Compose([
            T.ZNormalization(masking_method=normalization),
        ])
    }

    return data_transform