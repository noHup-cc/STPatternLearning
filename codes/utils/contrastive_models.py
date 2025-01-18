#!/usr/bin/env python

import copy
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from configparser import ConfigParser
if __name__ == '__main__':
    # Read config
    cfg = ConfigParser() 
    CONFIG_PATH = "../../config/config.ini"
    cfg.read(CONFIG_PATH) 
    CODES_DIR = cfg.get('codes', 'codes_dir')
    sys.path.append(CODES_DIR)
import utils.training_functions as training_functions

"""
@Article{chen2021mocov3,
  author  = {Xinlei Chen* and Saining Xie* and Kaiming He},
  title   = {An Empirical Study of Training Self-Supervised Vision Transformers},
  journal = {arXiv preprint arXiv:2104.02057},
  year    = {2021},
}
"""

def _build_mlp(num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
    """
    Build MLP with specified architecture.
    Implementation based on MoCo v3 paper.
    """
    mlp = []
    for l in range(num_layers):
        dim1 = input_dim if l == 0 else mlp_dim
        dim2 = output_dim if l == num_layers - 1 else mlp_dim

        mlp.append(nn.Linear(dim1, dim2, bias=False))
        if l < num_layers - 1:
            mlp.append(nn.BatchNorm1d(dim2))
            mlp.append(nn.ReLU(inplace=True))
        elif last_bn:
            # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
            # for simplicity, we further removed gamma in BN
            mlp.append(nn.BatchNorm1d(dim2, affine=False))
            
    return nn.Sequential(*mlp)

def build_projector_mlps(input_dim=None, dim=None, mlp_dim=None):
    """Build projector MLP with standard architecture."""
    hidden_dim = input_dim
    return _build_mlp(3, hidden_dim, mlp_dim, dim)

class SiameseNetwork(nn.Module):
    """Siamese network implementation with Swin Transformer backbone."""
    
    def __init__(self, pretrained_model, config=None):
        super(SiameseNetwork, self).__init__()
        self.config = config
        self.swin_22k = pretrained_model
        
        # Build projector
        projection_input_dim = 1536
        self.swin_22k.head = build_projector_mlps(
            input_dim=projection_input_dim,
            dim=self.config.MODEL.PROJECTION_DIM,
            mlp_dim=4096
        )
    
    def get_swin_22k_features(self, x):
        """Get features from the backbone before projection."""
        inps, outs = [], []
        def layer_hook(module, input, output):
            inps.append(input)
            outs.append(output)
            
        hook = self.swin_22k.head.register_forward_hook(layer_hook)
        output = self.swin_22k(x)
        hook.remove()
        
        return inps
    
    def forward_once(self, x):
        """Forward pass for one branch of Siamese network."""
        return self.swin_22k(x)
    
    def forward(self, img0, img1, label):
        """Forward pass for both branches."""
        output0 = self.forward_once(img0)
        output1 = self.forward_once(img1)
        return self.contrastive_loss(output0, output1, label)
    
    def contrastive_loss(self, output0, output1, label):
        """Calculate contrastive loss between outputs."""
        euclidean_distance = F.pairwise_distance(output0, output1, keepdim=True)
        loss_contrastive = torch.mean((label) * torch.pow(euclidean_distance, 2) +
                                    (1-label) * torch.pow(torch.clamp(self.config.MODEL.CONTRASTIVE_MARGIN - euclidean_distance, min=0.0), 2))
        
        return loss_contrastive

def get_model(feature_type=None, resume_model_path=None, config=None):
    """Create and load model based on feature type."""
    if feature_type == 'resnet':
        checkpoint = torch.load(resume_model_path, map_location='cpu')
        model = timm.create_model('resnet50', pretrained=False, num_classes=checkpoint['num_classes'])
        model.load_state_dict(checkpoint['state_dict'], strict=True)
    elif feature_type == 'swin_22k':
        model = timm.create_model('swin_large_patch4_window7_224_in22k', pretrained=True)
    elif feature_type == 'siam_swin_22k':
        if os.path.isfile(resume_model_path):
            print(f'Found pretrained model at {resume_model_path}, loading...')
            pretrained_model = timm.create_model('swin_large_patch4_window7_224_in22k', pretrained=True)
            checkpoint = torch.load(resume_model_path, map_location='cpu')
            model = SiameseNetwork(pretrained_model, config=config)
            model.load_state_dict(checkpoint['model'], strict=False)
            del checkpoint
            torch.cuda.empty_cache()
            print(model)
        else:
            print('model does not exist ...')
    else:
        print('feature_type does not exist')
        
    model.eval()
    if torch.cuda.device_count() > 0:
        model.cuda()
        
    return model

def get_feature_output_name(feature_type=None, is_projection=False, projection_output_dim=256, is_local=False, suffix=None):
    """Generate feature output name based on configuration."""
    if feature_type == 'swin_22k' or feature_type == 'resnet':
        print('output feature name :' + str(feature_type))
        output_feature_type = feature_type
    elif feature_type == 'siam_swin_22k':
        if is_projection == True:
            output_feature_type = feature_type + '_dim2_' + str(projection_output_dim)
        else:
            output_feature_type = feature_type + '_dim2_' + str(projection_output_dim) + '_dim1_1536'
            
        if is_local == True:
            output_feature_type = output_feature_type + '_local'
    else:
        print('get_feature_output_name() -> error')
        
    if suffix != None:
        output_feature_type = output_feature_type + suffix
        print('output_feature_type:' + str(output_feature_type))
            
    return output_feature_type

def extract_features(feature_type, img_path, model, n_gpu=0, is_projection=False):
    """Extract features from image using specified model."""
    img_array = training_functions.image_to_array(img_path)
    img_array = torch.unsqueeze(img_array, 0)
    if n_gpu > 0:
        img_array = img_array.cuda()
        
    if feature_type == 'resnet':
        def get_resnet_features(model, x):
            inps, outs = [], []
            def layer_hook(module, input, output):
                inps.append(input)
                outs.append(output)
            hook = model.fc.register_forward_hook(layer_hook)
            output = model(x)
            hook.remove()
            return inps
            
        features = get_resnet_features(model, img_array)
        features = features[0][0].cpu().detach().numpy()
    elif feature_type == 'swin_22k':
        def get_swin_22k_features(model, x):
            inps, outs = [], []
            def layer_hook(module, input, output):
                inps.append(input)
                outs.append(output)
            hook = model.head.register_forward_hook(layer_hook)
            output = model(x)
            hook.remove()
            return inps
            
        features = get_swin_22k_features(model, img_array)
        features = features[0][0].cpu().detach().numpy()
    elif feature_type == 'siam_swin_22k':
        if is_projection == True:
            features = model.forward_once(img_array).cpu().detach().numpy()
        else:
            features = model.get_swin_22k_features(img_array)[0][0].cpu().detach().numpy()
                
    return features