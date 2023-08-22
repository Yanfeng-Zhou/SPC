import numpy as np
import torchio as tio

def dataset_cfg(dataet_name):

    config = {
        'P-CT':
            {
                'IN_CHANNELS': 1,
                'NUM_CLASSES': 2,
                'NORMALIZE': tio.ZNormalization.mean,
                'PATCH_SIZE': (96, 96, 96),
                'PATCH_OVERLAP': (80, 80, 80),
                'FORMAT': '.nii',
                'NUM_SAMPLE_TRAIN': 4,
                'NUM_SAMPLE_VAL': 8,
                'QUEUE_LENGTH': 48
            },
        'LA':
            {
                'IN_CHANNELS': 1,
                'NUM_CLASSES': 2,
                'NORMALIZE': tio.ZNormalization.mean,
                'PATCH_SIZE': (112, 112, 80),
                'PATCH_OVERLAP': (94, 94, 76),
                'FORMAT': '.nrrd',
                'NUM_SAMPLE_TRAIN': 4,
                'NUM_SAMPLE_VAL': 8,
                'QUEUE_LENGTH': 48
            },
    }

    return config[dataet_name]
