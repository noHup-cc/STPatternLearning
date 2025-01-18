#!/usr/bin/env python

import os
import time
import datetime
import logging
import functools
import torch
import torch.optim as optim
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.scheduler.cosine_lr import CosineLRScheduler

"""
@article{xie2021moby,
  title={Self-Supervised Learning with Swin Transformers}, 
  author={Zhenda Xie and Yutong Lin and Zhuliang Yao and Zheng Zhang and Qi Dai and Yue Cao and Han Hu},
  journal={arXiv preprint arXiv:2105.04553},
  year={2021}
}
"""

def image_to_array(img_path, is_transform=False):
    """Convert image to tensor array with optional transformation."""
    img = Image.open(img_path).convert('RGB')
    
    if not is_transform:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        ])
        return transform(img)
    
    return img

def combine_param(param_list):
    """Combine parameters with and without weight decay."""
    param_has_decay = [p for param in param_list for p in param[0]['params']]
    param_no_decay = [p for param in param_list for p in param[1]['params']]
    
    return [
        {'params': param_has_decay},
        {'params': param_no_decay, 'weight_decay': 0.}
    ]

def set_weight_decay(model, skip_list=(), skip_keywords=()):
    """Set weight decay parameters based on layer types and names."""
    has_decay, no_decay = [], []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        if (len(param.shape) == 1 or name.endswith(".bias") or 
            name in skip_list or check_keywords_in_name(name, skip_keywords)):
            no_decay.append(param)
            print(f"{name} has no weight decay")
        else:
            has_decay.append(param)
            
    return [
        {'params': has_decay},
        {'params': no_decay, 'weight_decay': 0.}
    ]

def check_keywords_in_name(name, keywords=()):
    """Check if any keyword exists in parameter name."""
    return any(keyword in name for keyword in keywords)

def build_optimizer(config, model_list=[]):
    """Build optimizer with appropriate weight decay settings."""
    param_list = []
    
    for model in model_list:
        skip = getattr(model, 'no_weight_decay', lambda: set())()
        skip_keywords = getattr(model, 'no_weight_decay_keywords', lambda: set())()
        params = set_weight_decay(model, skip, skip_keywords)
        param_list.append(params)
    
    parameters = combine_param(param_list)
    opt_name = config.TRAIN.OPTIMIZER.NAME.lower()
    
    if opt_name == 'sgd':
        return optim.SGD(
            parameters,
            momentum=config.TRAIN.OPTIMIZER.MOMENTUM,
            nesterov=True,
            lr=config.TRAIN.BASE_LR,
            weight_decay=config.TRAIN.WEIGHT_DECAY
        )
    elif opt_name == 'adamw':
        return optim.AdamW(
            parameters,
            eps=config.TRAIN.OPTIMIZER.EPS,
            betas=config.TRAIN.OPTIMIZER.BETAS,
            lr=config.TRAIN.BASE_LR,
            weight_decay=config.TRAIN.WEIGHT_DECAY
        )
    
    raise ValueError(f"Unsupported optimizer: {opt_name}")

def build_scheduler(config, optimizer, n_iter_per_epoch):
    """Build learning rate scheduler."""
    num_steps = int(config.TRAIN.EPOCHS * n_iter_per_epoch)
    warmup_steps = int(config.TRAIN.WARMUP_EPOCHS * n_iter_per_epoch)
    
    if config.TRAIN.LR_SCHEDULER.NAME == 'cosine':
        return CosineLRScheduler(
            optimizer,
            t_initial=num_steps,
            lr_min=config.TRAIN.MIN_LR,
            warmup_lr_init=config.TRAIN.WARMUP_LR,
            warmup_t=warmup_steps,
            cycle_limit=1,
            t_in_epochs=False
        )
    
    raise ValueError("Unsupported scheduler type")

def adjust_training_parameters(config):
    """Adjust learning rates based on batch size and accumulation steps."""
    batch_scale = config.DATA.BATCH_SIZE / 512.0
    accum_scale = config.TRAIN.ACCUMULATION_STEPS if config.TRAIN.ACCUMULATION_STEPS > 1 else 1
    
    total_scale = batch_scale * accum_scale
    
    config.defrost()
    config.TRAIN.BASE_LR *= total_scale
    config.TRAIN.WARMUP_LR *= total_scale
    config.TRAIN.MIN_LR *= total_scale
    config.freeze()
    
    return config

class AverageMeter:
    """Compute and store the average and current value."""
    
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter:
    """Display training progress."""
    
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
    
    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
    
    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

@functools.lru_cache()
def create_logger(output_dir, name=''):
    """Create and configure logger."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    
    handler = logging.FileHandler(
        os.path.join(output_dir, 'log.txt'),
        mode='a'
    )
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(handler)
    
    return logger

def save_checkpoint(config, epoch, model, avg_train_loss, avg_val_loss,
                   optimizer, lr_scheduler, logger, scaler):
    """Save model checkpoint."""
    save_state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'avg_train_loss': avg_train_loss,
        'avg_val_loss': avg_val_loss,
        'epoch': epoch,
        'config': config
    }
    
    if config.IS_AMP:
        save_state['scaler'] = scaler.state_dict()
        logger.info("Saving scaler state...")
    
    save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth')
    checkpoint_path = os.path.join(config.OUTPUT, 'checkpoint.pth')
    
    logger.info(f"Saving checkpoint to {save_path}...")
    torch.save(save_state, save_path)
    torch.save(save_state, checkpoint_path)
    logger.info("Checkpoint saved successfully!")

def load_checkpoint(config, model, optimizer, lr_scheduler, logger, scaler):
    """Load model checkpoint."""
    logger.info(f"Resuming from {config.MODEL.RESUME}...")
    
    if config.MODEL.RESUME.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.MODEL.RESUME, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
    
    # Load model and training state
    model.load_state_dict(checkpoint['model'], strict=False)
    
    if all(k in checkpoint for k in ('optimizer', 'lr_scheduler', 'epoch')):
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        
        config.defrost()
        config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
        config.freeze()
        
        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    if 'scaler' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler'])
        logger.info("Loaded scaler state")
    
    del checkpoint
    torch.cuda.empty_cache()

def auto_resume_helper(output_dir):
    """Find latest checkpoint for auto-resume."""
    if os.path.exists(os.path.join(output_dir, 'checkpoint.pth')):
        return os.path.join(output_dir, 'checkpoint.pth')
    
    checkpoints = [ckpt for ckpt in os.listdir(output_dir) if ckpt.endswith('pth')]
    print(f"Found checkpoints in {output_dir}: {checkpoints}")
    
    if checkpoints:
        latest = max(
            [os.path.join(output_dir, d) for d in checkpoints],
            key=os.path.getmtime
        )
        print(f"Latest checkpoint: {latest}")
        return latest
    
    return None

class TrainWrapper:
    """Base class for training neural networks."""
    
    def __init__(self, model, config=None, data_loader_train=None, data_loader_val=None):
        if not data_loader_train or not data_loader_val:
            raise ValueError("Data loaders cannot be None")
            
        self.model = model.cuda()
        self.config = config
        self.data_loader_train = data_loader_train
        self.data_loader_val = data_loader_val
        
        # Setup logging
        log_folder = os.path.join(self.config.OUTPUT, 'log')
        tensorboard_folder = os.path.join(
            self.config.OUTPUT, 'tensorboard_log',
            datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        )
        os.makedirs(log_folder, exist_ok=True)
        os.makedirs(tensorboard_folder, exist_ok=True)
        
        self.logger = create_logger(log_folder, f"{self.config.MODEL.TYPE}")
        self.summary_writer = SummaryWriter(tensorboard_folder, purge_step=1)
        
        self.optimizer = None
        self.lr_scheduler = None
        self.scaler = None
    
    def training_loop(self):
        """Main training loop."""
        self.optimizer, self.lr_scheduler = self.configure_optimizers()
        self.scaler = torch.cuda.amp.GradScaler()
        
        # Handle auto-resume
        if self.config.TRAIN.AUTO_RESUME:
            resume_file = auto_resume_helper(self.config.OUTPUT)
            if resume_file:
                if self.config.MODEL.RESUME:
                    self.logger.warning(
                        f"Auto-resume changing resume file from "
                        f"{self.config.MODEL.RESUME} to {resume_file}"
                    )
                self.config.defrost()
                self.config.MODEL.RESUME = resume_file
                self.config.freeze()
                self.logger.info(f'Auto resuming from {resume_file}')
                
                load_checkpoint(
                    self.config, self.model, self.optimizer,
                    self.lr_scheduler, self.logger, self.scaler
                )
            else:
                raise RuntimeError(
                    f"No checkpoint found in {self.config.OUTPUT} for auto-resume"
                )
        
        # Training epochs
        for epoch in range(self.config.TRAIN.START_EPOCH, self.config.TRAIN.EPOCHS):
            avg_train_loss = self.train_one_epoch(epoch)
            avg_val_loss = self.validate_one_epoch(epoch)
            
            # Log metrics
            self.summary_writer.add_scalar("epoch_loss/train", avg_train_loss, epoch)
            self.summary_writer.add_scalar("epoch_loss/val", avg_val_loss, epoch)
            
            # Save checkpoint
            if epoch % self.config.SAVE_FREQ == 0 or epoch == (self.config.TRAIN.EPOCHS - 1):
                save_checkpoint(
                    self.config, epoch, self.model,
                    avg_train_loss, avg_val_loss,
                    self.optimizer, self.lr_scheduler,
                    self.logger, self.scaler
                )
    
    def train_one_epoch(self, epoch):
        """Training loop for one epoch."""
        self.model.train()
        self.optimizer.zero_grad()
        
        num_steps = len(self.data_loader_train)
        batch_time = AverageMeter('Train batch time', ':6.3f')
        loss_meter = AverageMeter('Train batch loss', ':.4e')
        
        progress = ProgressMeter(
            num_steps,
            [batch_time, loss_meter],
            prefix=f"Epoch: [{epoch}] (training)"
        )
        
        start = time.time()
        end = time.time()
        
        for batch_idx, batch in enumerate(self.data_loader_train):
            # Handle gradient accumulation
            if self.config.TRAIN.ACCUMULATION_STEPS >= 1:
                loss = self._compute_loss(batch_idx, batch, epoch)
                loss = loss / self.config.TRAIN.ACCUMULATION_STEPS
                
                # Update metrics
                loss_meter.update(loss.item(), self.config.DATA.BATCH_SIZE)
                self.summary_writer.add_scalar(
                    "batch_loss/train",
                    loss.item(),
                    epoch * num_steps + batch_idx
                )
                
                # Update gradients
                if self.config.IS_AMP:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
            
            # Update parameters if accumulation complete
            if (batch_idx + 1) % self.config.TRAIN.ACCUMULATION_STEPS == 0:
                self._update_parameters()
                self._update_learning_rate(epoch, num_steps, batch_idx)
            
            # Measure elapsed time
            batch_time.update(time.time() - end)

            # Log progress
            if batch_idx % self.config.PRINT_FREQ == 0:
                progress.display(batch_idx)
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                self.logger.info(
                    f'Train: [{epoch}/{self.config.TRAIN.EPOCHS}][{batch_idx}/{num_steps}]\t'
                    f'Train batch time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                    f'Train batch loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                    f'Train batch mem {memory_used:.0f}MB'
                )
            
            end = time.time()
        
        epoch_time = time.time() - start
        self.logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")
        
        return loss_meter.avg

    def _compute_loss(self, batch_idx, batch, epoch):
        """Compute loss with automatic mixed precision if enabled."""
        if self.config.IS_AMP:
            with torch.cuda.amp.autocast(True):
                return self.training_step(batch_idx, batch, epoch)
        return self.training_step(batch_idx, batch, epoch)

    def _update_parameters(self):
        """Update model parameters based on accumulated gradients."""
        if self.config.IS_AMP:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self.optimizer.zero_grad()

    def _update_learning_rate(self, epoch, num_steps, batch_idx):
        """Update learning rate and log to tensorboard."""
        self.lr_scheduler.step_update(epoch * num_steps + batch_idx)
        lr = self.optimizer.param_groups[0]['lr']
        self.summary_writer.add_scalar(
            "learning_rate",
            lr,
            epoch * num_steps + batch_idx
        )

    @torch.no_grad()
    def validate_one_epoch(self, epoch):
        """Validation loop for one epoch."""
        self.model.eval()
        num_steps = len(self.data_loader_val)
        
        batch_time = AverageMeter('Validation batch time', ':6.3f')
        loss_meter = AverageMeter('Validation batch loss', ':.4e')
        
        progress = ProgressMeter(
            num_steps,
            [batch_time, loss_meter],
            prefix=f"Epoch: [{epoch}] (validation)"
        )
        
        end = time.time()
        
        for batch_idx, batch in enumerate(self.data_loader_val):
            # Compute validation loss
            loss = self.validation_step(batch_idx, batch, epoch)
            loss_meter.update(loss.item(), self.config.DATA.BATCH_SIZE)
            
            # Log metrics
            self.summary_writer.add_scalar(
                "batch_loss/val",
                loss.item(),
                epoch * num_steps + batch_idx
            )
            
            # Measure elapsed time
            batch_time.update(time.time() - end)
            
            # Log progress
            if batch_idx % self.config.PRINT_FREQ == 0:
                progress.display(batch_idx)
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                self.logger.info(
                    f'Batch validation time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    f'Batch validation loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                    f'Batch validation mem {memory_used:.0f}MB'
                )
            
            end = time.time()
        
        return loss_meter.avg

    def training_step(self, batch_idx, batch, epoch):
        """To be implemented by subclasses."""
        raise NotImplementedError

    def validation_step(self, batch_idx, batch, epoch):
        """To be implemented by subclasses."""
        raise NotImplementedError

    def configure_optimizers(self):
        """To be implemented by subclasses."""
        raise NotImplementedError