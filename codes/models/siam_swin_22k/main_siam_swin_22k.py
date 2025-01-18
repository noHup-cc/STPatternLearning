#!/usr/bin/env python

import os
import sys
import torch
import numpy as np
import timm
from argparse import ArgumentParser
from configparser import ConfigParser
if __name__ == '__main__':
    # Read config
    cfg = ConfigParser() 
    CONFIG_PATH = "../../../config/config.ini"
    cfg.read(CONFIG_PATH) 
    CODES_DIR = cfg.get('codes', 'codes_dir')
    sys.path.append(CODES_DIR)
from models.config import get_config
import utils.training_functions as training_functions
import utils.data_wrapper as data_wrapper
import utils.contrastive_models as contrastive_models

class SiamTrainWrapper(training_functions.TrainWrapper):
    """Wrapper class for Siamese network training."""
    
    def __init__(self, model, config=None, data_loader_train=None, data_loader_val=None):
        """Initialize the training wrapper."""
        super().__init__(
            model=model,
            config=config,
            data_loader_train=data_loader_train,
            data_loader_val=data_loader_val
        )
    
    def training_step(self, batch_idx, batch, epoch):
        """Execute single training step."""
        img0, img1, label = [x.cuda(non_blocking=True) for x in batch]
        return self.model.forward(img0, img1, label)
    
    def validation_step(self, batch_idx, batch, epoch):
        """Execute single validation step."""
        return self.training_step(batch_idx, batch, epoch)
    
    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        optimizer = training_functions.build_optimizer(self.config, [self.model])
        lr_scheduler = training_functions.build_scheduler(
            self.config, optimizer, len(self.data_loader_train)
        )
        return optimizer, lr_scheduler

def setup_argument_parser():
    """Configure command line arguments."""
    parser = ArgumentParser()
    parser.add_argument(
        "--cfg",
        type=str,
        default="models/siam_swin_22k/configs/siam_swin_22k.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--pretrain",
        action='store_true',
        help="Whether to use pretrained model or not"
    )
    parser.add_argument(
        "--is_amp",
        action='store_true',
        help="Enable automatic mixed precision"
    )
    parser.add_argument(
        "--total_dataset",
        type=str,
        default='embryo_registration/data/samples/datasets/total_datasets/embryoABC/total_dataset_embryoABC.csv',
        help="Path to total dataset"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default='embryo_registration/codes/models/trained_models/siamese_contra_swin_22k_14_01_2023_train_full_val_full_dim2_256',
        help="Path to save checkpoints"
    )
    parser.add_argument(
        "--is_resume_training",
        action='store_true',
        help="Whether to resume training from checkpoint"
    )
    parser.add_argument(
        "--resume_training_path",
        type=str,
        default="embryo_registration/...",
        help="Path to checkpoint for resuming training"
    )
    return parser

def setup_environment(cfg):
    """Setup training environment and directories."""
    # Setup checkpoint directory
    checkpoint_folder = os.path.join(cfg.get('codes', 'codes_dir'), 'models/torch_checkpoints')
    os.makedirs(checkpoint_folder, exist_ok=True)
    torch.hub.set_dir(checkpoint_folder)
    
    # Set random seeds for reproducibility
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)
    
    # Report GPU availability
    print(f'Available GPUs: {torch.cuda.device_count()}')

def load_model(pretrain):
    """Load Swin Transformer model."""
    model_name = 'swin_large_patch4_window7_224_in22k'
    pretrain_status = "with" if pretrain else "without"
    print(f'Loading {model_name} {pretrain_status} pretraining...')
    
    return timm.create_model(model_name, pretrained=pretrain)

def prepare_training(args, config, config_path):
    """Prepare datasets and model for training."""
    # Load dataset
    dataset_wrapper = data_wrapper.SiamDataWrapper(
        config, args.total_dataset,
        col_list=['embryoType', 'info', 'Class'],
        config_path=config_path,
        training_data_use_proportion=1.0,
        validation_data_use_proportion=1.0
    )
    
    train_dataset, val_dataset = dataset_wrapper.build_dataset()
    data_loaders = data_wrapper.build_loader(
        dataset_wrapper.config,
        train_dataset,
        val_dataset
    )
    
    # Build model
    pretrained_model = load_model(args.pretrain)
    model = contrastive_models.SiameseNetwork(pretrained_model, config=config)
    
    return model, data_loaders

def main():
    """Main function to run training process."""    
    # Setup environment
    setup_environment(cfg)
    
    # Parse arguments and update paths
    parser = setup_argument_parser()
    args = parser.parse_args()
    args.cfg = os.path.join(cfg.get('codes', 'codes_dir'), args.cfg)
    args.total_dataset = os.path.join(cfg.get('basic', 'home_dir'), args.total_dataset)
    args.checkpoint_path = os.path.join(cfg.get('basic', 'home_dir'), args.checkpoint_path)
    
    print(f'Configuration file: {args.cfg}')
    
    # Get and adjust configuration
    config = get_config(args)
    config = training_functions.adjust_training_parameters(config)
    
    # Prepare for training
    model, (data_loader_train, data_loader_val) = prepare_training(args, config, CONFIG_PATH)
    
    # Initialize trainer and start training
    trainer = SiamTrainWrapper(
        model,
        config=config,
        data_loader_train=data_loader_train,
        data_loader_val=data_loader_val
    )
    trainer.training_loop()

if __name__ == '__main__':
    main()