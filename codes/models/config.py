#! /usr/bin/env python
import os
import yaml
from yacs.config import CfgNode as CN


"""
@article{xie2021moby,
  title={Self-Supervised Learning with Swin Transformers}, 
  author={Zhenda Xie and Yutong Lin and Zhuliang Yao and Zheng Zhang and Qi Dai and Yue Cao and Han Hu},
  journal={arXiv preprint arXiv:2105.04553},
  year={2021}
}
"""

_C = CN()


# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model type
_C.MODEL.TYPE = 'swin'
# Model name
# _C.MODEL.NAME = 'swin_tiny_patch4_window7_224'
# Checkpoint to resume, could be overwritten by command line argument
_C.MODEL.RESUME = ''
# Number of classes, overwritten in data preparation
_C.MODEL.PROJECTION_DIM = 256
_C.MODEL.CONTRASTIVE_MARGIN = 1.25
_C.MODEL.TRIPLET_MARGIN = 0.2

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Batch size for a single GPU, could be overwritten by command line argument
_C.DATA.BATCH_SIZE = 64
# Path to dataset, could be overwritten by command line argument
# _C.DATA.DATA_PATH = ''
# Dataset name
# _C.DATA.DATASET = 'imagenet'
# Input image size
# _C.DATA.IMG_SIZE = 224
# Interpolation to resize image (random, bilinear, bicubic)
# _C.DATA.INTERPOLATION = 'bicubic'
# Use zipped dataset instead of folder dataset
# could be overwritten by command line argument
# _C.DATA.ZIP_MODE = False
# Cache Data in Memory, could be overwritten by command line argument
# _C.DATA.CACHE_MODE = 'part'
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.DATA.PIN_MEMORY = True
# Number of data loading threads
_C.DATA.NUM_WORKERS = 8

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 300
_C.TRAIN.NUM_GPUS = 1
_C.TRAIN.NUM_PATIENCES = 5
_C.TRAIN.WARMUP_EPOCHS = 20
_C.TRAIN.WEIGHT_DECAY = 0.05
_C.TRAIN.BASE_LR = 5e-4
_C.TRAIN.WARMUP_LR = 5e-7
_C.TRAIN.MIN_LR = 5e-6
# Clip gradient norm
# _C.TRAIN.CLIP_GRAD = 5.0
# Auto resume from latest checkpoint
_C.TRAIN.AUTO_RESUME = True
# Gradient accumulation steps
# could be overwritten by command line argument
_C.TRAIN.ACCUMULATION_STEPS = 0
# Whether to use gradient checkpointing to save memory
# could be overwritten by command line argument
_C.TRAIN.USE_CHECKPOINT = False

# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'
# Epoch interval to decay LR, used in StepLRScheduler
# _C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
# LR decay rate, used in StepLRScheduler
# _C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adamw'
# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
# Mixed precision opt level, if O0, no amp is used ('O0', 'O1', 'O2')
# overwritten by command line argument
# _C.AMP_OPT_LEVEL = ''
_C.IS_AMP = False
# Path to output folder, overwritten by command line argument
_C.OUTPUT = ''
# Tag of experiment, overwritten by command line argument
# _C.TAG = 'default'
# Frequency to save checkpoint
_C.SAVE_FREQ = 1
# Frequency to logging info
_C.PRINT_FREQ = 10

# -----------------------------------------------------------------------------
# Augmentation settings
# -----------------------------------------------------------------------------
_C.AUG = CN()
# Color jitter factor
# _C.AUG.COLOR_JITTER = 0.4
# Use AutoAugment policy. "v0" or "original"
# _C.AUG.AUTO_AUGMENT = 'rand-m9-mstd0.5-inc1'
# Random erase prob
# _C.AUG.REPROB = 0.25
# Random erase mode
# _C.AUG.REMODE = 'pixel'
# Random erase count
# _C.AUG.RECOUNT = 1
# Mixup alpha, mixup enabled if > 0
# _C.AUG.MIXUP = 0.8
# Cutmix alpha, cutmix enabled if > 0
# _C.AUG.CUTMIX = 1.0
# Probability of performing mixup or cutmix when either/both is enabled
# _C.AUG.MIXUP_PROB = 1.0
# Probability of switching to cutmix when both mixup and cutmix enabled
# _C.AUG.MIXUP_SWITCH_PROB = 0.5
# How to apply mixup/cutmix params. Per "batch", "pair", or "elem"
# _C.AUG.MIXUP_MODE = 'batch'
# Self-Supervised Learning Augmentation
# _C.AUG.SSL_AUG = False
# SSL-Aug type
# _C.AUG.SSL_AUG_TYPE = 'byol'
# SSL-Aug crop
_C.AUG.SSL_AUG_CROP = 0.08
# Self-Supervised Learning Linear Evaluation Augmentation
# _C.AUG.SSL_LINEAR_AUG = False


def update_config_from_file(config, cfg_file):
    """
    Update configuration from a config file.
    
    Args:
        config: Configuration object (yacs CfgNode)
        cfg_file: Path to config file
        
    Returns:
        Updated config object
    """
    print(f'=> Merging config from {cfg_file}')
    
    config.defrost()
    config.merge_from_file(cfg_file)
    config.freeze()
    
    return config
    
def update_config(config, args, enable_args=True):
    """
    Update configuration from file and command line arguments.
    
    Args:
        config: Configuration object (yacs CfgNode)
        args: Command line arguments
        enable_args: Whether to enable command line argument updates
        
    Returns:
        Updated config object
    """
    # Update from config file
    config = update_config_from_file(config, args.cfg)
    
    if not enable_args:
        print('Only using config file arguments')
        return config
        
    # Update from command line args
    config.defrost()
    
    if args.checkpoint_path:
        config.MODEL.RESUME = args.checkpoint_path
        config.OUTPUT = args.checkpoint_path
        
    config.TRAIN.AUTO_RESUME = args.is_resume_training
    config.IS_AMP = args.is_amp
    
    config.freeze()
    return config

def get_config(args, enable_args=True):
    """
    Get configuration with default values and updates.
    
    Args:
        args: Command line arguments
        enable_args: Whether to enable command line argument updates
        
    Returns:
        Complete configuration object
    """
    # Clone default config to avoid modifying defaults
    config = _C.clone()
    
    # Update with provided arguments
    config = update_config(config, args, enable_args=enable_args)
    
    return config