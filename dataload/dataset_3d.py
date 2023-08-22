import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import numpy as np
import torchio as tio
import SimpleITK as sitk
from torchio.data import UniformSampler, LabelSampler


class dataset_it(Dataset):
    def __init__(self, data_dir, transform_1, queue_length=20, samples_per_volume=5, patch_size=128, num_workers=8, shuffle_subjects=True, shuffle_patches=True, sup=True, num_images=None):
        super(dataset_it, self).__init__()

        self.subjects_1 = []

        image_dir_1 = data_dir + '/image'
        if sup:
            mask_dir = data_dir + '/mask'

        for i in os.listdir(image_dir_1):
            image_path_1 = os.path.join(image_dir_1, i)
            if sup:
                mask_path = os.path.join(mask_dir, i)
                subject_1 = tio.Subject(image=tio.ScalarImage(image_path_1), mask=tio.LabelMap(mask_path), ID=i)
            else:
                subject_1 = tio.Subject(image=tio.ScalarImage(image_path_1), ID=i)

            self.subjects_1.append(subject_1)

        if num_images is not None:
            len_img_paths = len(self.subjects_1)
            quotient = num_images // len_img_paths
            remainder = num_images % len_img_paths

            if num_images <= len_img_paths:
                self.subjects_1 = self.subjects_1[:num_images]
            else:
                rand_indices = torch.randperm(len_img_paths).tolist()
                new_indices = rand_indices[:remainder]

                self.subjects_1 = self.subjects_1 * quotient
                self.subjects_1 += [self.subjects_1[i] for i in new_indices]

        self.dataset_1 = tio.SubjectsDataset(self.subjects_1, transform=transform_1)

        self.queue_train_set_1 = tio.Queue(
            subjects_dataset=self.dataset_1,
            max_length=queue_length,
            samples_per_volume=samples_per_volume,
            sampler=UniformSampler(patch_size),
            # sampler=LabelSampler(patch_size),
            num_workers=num_workers,
            shuffle_subjects=shuffle_subjects,
            shuffle_patches=shuffle_patches
        )
