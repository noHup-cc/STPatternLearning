#!/usr/bin/env python

import sys
import re
import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageFilter, ImageOps
from configparser import ConfigParser
if __name__ == '__main__':
    # Read config
    cfg = ConfigParser() 
    CONFIG_PATH = "../../config/config.ini"
    cfg.read(CONFIG_PATH) 
    CODES_DIR = cfg.get('codes', 'codes_dir')
    sys.path.append(CODES_DIR)
import utils.data_process_mouse_embryo as data_process_mouse_embryo
import utils.training_functions as training_functions

class GaussianBlur:
    """Gaussian blur augmentation from SimCLR: https://arxiv.org/abs/2002.05709"""
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        return x.filter(ImageFilter.GaussianBlur(radius=sigma))

class Solarize:
    """Solarize augmentation from BYOL: https://arxiv.org/abs/2006.07733"""
    def __call__(self, x):
        return ImageOps.solarize(x)

def divide_datasets(samples_data, training_data_use_proportion=1.0,
                   validation_data_use_proportion=1.0, test_data_use_proportion=0.0):
    """
    Divide dataset into training, validation, and optionally test sets.
    
    Args:
        samples_data: Input dataset
        training_data_use_proportion: Proportion of training data to use
        validation_data_use_proportion: Proportion of validation data to use
        test_data_use_proportion: Proportion of test data to use
    """
    # Divide datasets
    training_data, validation_data, test_data = data_process_mouse_embryo.divide_datasets(samples_data.copy())
    
    # Verify division
    if (len(training_data) + len(validation_data) + len(test_data)) != len(samples_data):
        raise ValueError("Divided datasets size doesn't match original dataset")
        
    # Sample data based on proportions
    training_data = training_data.sample(frac=training_data_use_proportion, random_state=25)
    validation_data = validation_data.sample(frac=validation_data_use_proportion, random_state=25)
    test_data = test_data.sample(frac=test_data_use_proportion, random_state=25)
    
    return (training_data, validation_data, test_data) if test_data_use_proportion > 0 else (training_data, validation_data)

def build_loader(config, train_dataset, val_dataset):
    """Build data loaders for training and validation."""
    if not train_dataset or not val_dataset:
        raise ValueError("Train or validation dataset is None")

    data_loader_train = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        pin_memory=config.DATA.PIN_MEMORY,
        num_workers=config.DATA.NUM_WORKERS
    )
    
    data_loader_val = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        pin_memory=config.DATA.PIN_MEMORY,
        num_workers=config.DATA.NUM_WORKERS
    )
    
    return data_loader_train, data_loader_val

def build_transform(config):
    """Build data transformations for training and validation."""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    # First augmentation pipeline
    augmentation1 = [
        transforms.RandomResizedCrop(224, scale=(config.AUG.SSL_AUG_CROP, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=1.0),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]
    
    # Second augmentation pipeline
    augmentation2 = [
        transforms.RandomResizedCrop(224, scale=(config.AUG.SSL_AUG_CROP, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.1),
        transforms.RandomApply([Solarize()], p=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]

    transform_train = [transforms.Compose(augmentation1), transforms.Compose(augmentation2)]
    transform_val = [transforms.Compose(augmentation1), transforms.Compose(augmentation2)]
    
    return transform_train, transform_val

class SiamDataset(Dataset):
    """Dataset class for Siamese network training."""
    
    def __init__(self, data_info, header=True, transform=[None, None]):
        self.header = header
        self.data_info = data_info
        self.transform1 = transform[0]
        self.transform2 = transform[1]
        self.regex = re.compile(r'\d+')

    def __len__(self):
        return len(self.data_info) - 1 if self.header else len(self.data_info)

    def __getitem__(self, idx):
        """Get transformed image pair and label."""
        data_index = idx + 1 if self.header else idx
        
        # Load and transform images
        if self.transform1 and self.transform2:
            original_slice = training_functions.image_to_array(
                self.data_info[data_index][2], is_transform=True)
            original_slice = self.transform1(original_slice)
            
            second_slice = training_functions.image_to_array(
                self.data_info[data_index][3], is_transform=True)
            second_slice = self.transform2(second_slice)
        else:
            original_slice = training_functions.image_to_array(
                self.data_info[data_index][2])
            second_slice = training_functions.image_to_array(
                self.data_info[data_index][3])
        
        # Process label
        label = 0.0 if int(self.data_info[data_index][4]) == -1 else float(self.data_info[data_index][4])
        label = torch.tensor([label], dtype=torch.float32)
        
        return original_slice, second_slice, label

class SiamDataWrapper:
    """Wrapper class for handling Siamese network data preparation."""
    
    def __init__(self, config, sample_path, col_list=None, config_path=None,
                 training_data_use_proportion=1.0, validation_data_use_proportion=1.0):
        self.config = config
        self.config_path = config_path
        self.training_data_use_proportion = training_data_use_proportion
        self.validation_data_use_proportion = validation_data_use_proportion
        
        # Load and divide dataset
        self.samples_data = pd.read_csv(sample_path, usecols=col_list) if col_list else pd.read_csv(sample_path)
        self.training_data, self.validation_data = divide_datasets(
            self.samples_data,
            training_data_use_proportion=self.training_data_use_proportion,
            validation_data_use_proportion=self.validation_data_use_proportion
        )
    
    def get_sample_annotation(self, config_path=None, samples_data_list=None):
        """Generate sample annotations with image paths."""
        output_cols = ['embryoType', 'info', 'original_slice_image', 'second_slice_image', 'Class']
        sample_image_pairs = [output_cols]
        
        for sample in samples_data_list:
            embryo_type, info, label = sample[0], sample[1], sample[2]
            original_info, second_info = info.split('_')
            
            # Extract time and z values
            original_time, original_z = map(int, re.findall(r'\d+', original_info)[-2:])
            second_time, second_z = map(int, re.findall(r'\d+', second_info)[-2:])
            
            # Get image paths
            original_path = data_process_mouse_embryo.get_embryo_slice_path(
                str(embryo_type), original_time, original_z, config_path=config_path)
            second_path = data_process_mouse_embryo.get_embryo_slice_path(
                str(embryo_type), second_time, second_z, config_path=config_path)
            
            sample_image_pairs.append([embryo_type, info, original_path, second_path, label])
            
        return sample_image_pairs
    
    def build_dataset(self):
        """Build training and validation datasets."""
        # Create sample annotations
        training_samples = self.get_sample_annotation(
            config_path=self.config_path,
            samples_data_list=self.training_data.values.tolist()
        )
        validation_samples = self.get_sample_annotation(
            config_path=self.config_path,
            samples_data_list=self.validation_data.values.tolist()
        )
        
        # Build transforms and datasets
        transform_train, transform_val = build_transform(self.config)
        train_dataset = SiamDataset(training_samples, transform=transform_train, header=True)
        val_dataset = SiamDataset(validation_samples, transform=transform_val, header=True)
        
        return train_dataset, val_dataset