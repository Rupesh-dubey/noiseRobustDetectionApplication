import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import timm
from timm import create_model
from timm.models.registry import register_model
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers.activations import Swish
from timm.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.efficientnet import efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7
from collections import OrderedDict
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model configuration
num_classes = 2
img_size = 456
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# PPM Module (exact copy from your original code)
class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim=512, pool_scales=(1, 2, 3, 6)):
        super(PPM, self).__init__()
        self.features = []
        for scale in pool_scales:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)
        self.conv_final = nn.Sequential(
            nn.Conv2d(in_dim + reduction_dim * len(pool_scales), reduction_dim,
                     kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(reduction_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
        
    def forward(self, x):
        input_size = x.size()
        ppm_out = [x]
        for pool_scale in self.features:
            ppm_out.append(nn.functional.interpolate(
                pool_scale(x),
                size=(input_size[2], input_size[3]),
                mode='bilinear', align_corners=False))
        ppm_out = torch.cat(ppm_out, 1)
        return self.conv_final(ppm_out)

# Configuration function (exact copy from your original code)
def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': 1.0, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': '', 'classifier': 'head',
        **kwargs
    }

# Default configurations for all models (exact copy from your original code)
default_cfgs = {
    'vit_small_efficientnet_b0': _cfg(),
    'vit_small_efficientnet_b1': _cfg(),
    'vit_small_efficientnet_b2': _cfg(),
    'vit_small_efficientnet_b3': _cfg(),
    'vit_small_efficientnet_b4': _cfg(),
    'vit_small_efficientnet_b5': _cfg(),
    'vit_small_efficientnet_b6': _cfg(),
    'vit_small_efficientnet_b7': _cfg(),
}

# MLP Module (exact copy from your original code)
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(drop)  # seems more common to have Transformer MLP drouput here?

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

# Attention Module (exact copy from your original code)
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.scale = 1. / dim ** 0.5
        self.num_heads = num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv[:, :, 0].transpose(1, 2), qkv[:, :, 1].transpose(1, 2), qkv[:, :, 2].transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        # FIXME support masking
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# Block Module (exact copy from your original code) - COMPLETE IMPLEMENTATION
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., act_layer=nn.GELU, drop=0., drop_path=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, attn_drop=drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, attn_mask=None):
        x = x + self.drop_path(self.attn(self.norm1(x), attn_mask=attn_mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

# PatchEmbed Module (exact copy from your original code)
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, flatten_channels_last=False):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        assert img_size[0] % patch_size[0] == 0, 'image height must be divisible by the patch height'
        assert img_size[1] % patch_size[1] == 0, 'image width must be divisible by the patch width'
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        patch_dim = in_chans * patch_size[0] * patch_size[1]
        self.img_size = img_size
        self.patch_size = patch_size
        self.flatten_channels_last = flatten_channels_last
        self.num_patches = num_patches

        self.proj = nn.Linear(patch_dim, embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        Ph, Pw = self.patch_size
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        if self.flatten_channels_last:
            # flatten patches with channels last like the paper (likely using TF)
            x = x.unfold(2, Ph, Ph).unfold(3, Pw, Pw).permute(0, 2, 3, 4, 5, 1).reshape(B, -1, Ph * Pw * C)
        else:
            x = x.permute(0, 2, 3, 1).unfold(1, Ph, Ph).unfold(2, Pw, Pw).reshape(B, -1, C * Ph * Pw)
        x = self.proj(x)
        return x

# HybridEmbed Module (exact copy from your original code)
class HybridEmbed(nn.Module):
    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            feature_dim = self.backbone.feature_info.channels()[-1]
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Linear(feature_dim, embed_dim)

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x

# Vision Transformer (exact copy from your original code)
class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., mlp_head=False, drop_rate=0., drop_path_rate=0.,
                 flatten_channels_last=False, hybrid_backbone=None, use_ppm=True,**kwargs):
        super().__init__()
        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
            # Get the number of channels from the backbone
            with torch.no_grad():
                training = hybrid_backbone.training
                if training:
                    hybrid_backbone.eval()
                o = hybrid_backbone(torch.zeros(1, in_chans, img_size, img_size))[-1]
                in_dim = o.shape[1]
                hybrid_backbone.train(training)
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
                flatten_channels_last=flatten_channels_last)
            in_dim = in_chans * patch_size * patch_size
        num_patches = self.patch_embed.num_patches
        # Add PPM module with correct input dimension
        self.use_ppm = use_ppm
        if use_ppm:
            self.ppm = PPM(in_dim=in_dim, reduction_dim=512)  # Use the detected input dimension
            self.ppm_proj = nn.Linear(512, embed_dim)
            
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, drop=drop_rate, drop_path=dpr[i])
            for i in range(depth)])

        self.norm = nn.LayerNorm(embed_dim)
        if mlp_head:
            self.head = Mlp(embed_dim, int(embed_dim * mlp_ratio), num_classes)
        else:
            # with a single Linear layer as head, the param count within rounding of paper
            self.head = nn.Linear(embed_dim, num_classes)

        # FIXME not quite sure what the proper weight init is supposed to be, 
        # normal / trunc normal w/ std == .02 similar to other Bert like transformers
        trunc_normal_(self.pos_embed, std=.02)  # embeddings same as weights?
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights) 
        self.use_ppm = use_ppm

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @property
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward(self, x):
        B = x.shape[0]
        
        if not self.use_ppm:
            x = self.patch_embed(x)
        else:
            backbone_features = self.patch_embed.backbone(x)[-1]
            ppm_features = self.ppm(backbone_features)
            x = ppm_features.flatten(2).transpose(1, 2)
            x = self.ppm_proj(x)
    
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed[:, :x.size(1), :]
    
        for blk in self.blocks: 
            x = blk(x)
    
        x = self.norm(x[:, 0])
        x = self.head(x)
        return x

# All registered model functions (exact copies from your original code)
@register_model
def vit_small_efficientnet_b0(pretrained=False, **kwargs):
    pretrained_backbone = kwargs.get('pretrained_backbone', True)
    backbone = efficientnet_b0(pretrained=pretrained_backbone, features_only=True, out_indices=[4])
    model = VisionTransformer(
        img_size=456, embed_dim=768, depth=8, num_heads=8, mlp_ratio=3,
        hybrid_backbone=backbone, use_ppm=True, num_classes=num_classes, **kwargs)
    model.default_cfg = default_cfgs['vit_small_efficientnet_b0']
    return model

@register_model
def vit_small_efficientnet_b1(pretrained=False, **kwargs):
    pretrained_backbone = kwargs.get('pretrained_backbone', True)
    backbone = efficientnet_b1(pretrained=pretrained_backbone, features_only=True, out_indices=[4])
    model = VisionTransformer(
        img_size=456, embed_dim=768, depth=8, num_heads=8, mlp_ratio=3,
        hybrid_backbone=backbone, use_ppm=True, num_classes=num_classes, **kwargs)
    model.default_cfg = default_cfgs['vit_small_efficientnet_b1']
    return model

@register_model
def vit_small_efficientnet_b2(pretrained=False, **kwargs):
    pretrained_backbone = kwargs.get('pretrained_backbone', True)
    backbone = efficientnet_b2(pretrained=pretrained_backbone, features_only=True, out_indices=[4])
    model = VisionTransformer(
        img_size=456, embed_dim=768, depth=8, num_heads=8, mlp_ratio=3,
        hybrid_backbone=backbone, use_ppm=True, num_classes=num_classes, **kwargs)
    model.default_cfg = default_cfgs['vit_small_efficientnet_b2']
    return model

@register_model
def vit_small_efficientnet_b3(pretrained=False, **kwargs):
    pretrained_backbone = kwargs.get('pretrained_backbone', True)
    backbone = efficientnet_b3(pretrained=pretrained_backbone, features_only=True, out_indices=[3])
    model = VisionTransformer(
        img_size=456, embed_dim=768, depth=8, num_heads=8, mlp_ratio=3,
        hybrid_backbone=backbone, use_ppm=True, num_classes=num_classes, **kwargs)
    model.default_cfg = default_cfgs['vit_small_efficientnet_b3']
    return model

@register_model
def vit_small_efficientnet_b4(pretrained=False, **kwargs):
    pretrained_backbone = kwargs.get('pretrained_backbone', True)
    backbone = efficientnet_b4(pretrained=pretrained_backbone, features_only=True, out_indices=[3])
    model = VisionTransformer(
        img_size=456, embed_dim=768, depth=8, num_heads=8, mlp_ratio=3,
        hybrid_backbone=backbone, use_ppm=True, num_classes=num_classes, **kwargs)
    model.default_cfg = default_cfgs['vit_small_efficientnet_b4']
    return model

@register_model
def vit_small_efficientnet_b5(pretrained=False, **kwargs):
    pretrained_backbone = kwargs.get('pretrained_backbone', True)
    backbone = efficientnet_b5(pretrained=pretrained_backbone, features_only=True, out_indices=[3])  # Changed out_indices to [3]
    model = VisionTransformer(
        img_size=456, embed_dim=768, depth=8, num_heads=8, mlp_ratio=3, 
        hybrid_backbone=backbone, use_ppm=True, num_classes=num_classes, **kwargs)
    model.default_cfg = default_cfgs['vit_small_efficientnet_b5']
    print(f"Backbone feature stages: {[f['stage'] for f in backbone.feature_info]}")
    return model

@register_model
def vit_small_efficientnet_b6(pretrained=False, **kwargs):
    pretrained_backbone = kwargs.get('pretrained_backbone', True)
    backbone = efficientnet_b6(pretrained=pretrained_backbone, features_only=True, out_indices=[3])
    model = VisionTransformer(
        img_size=456, embed_dim=768, depth=8, num_heads=8, mlp_ratio=3,
        hybrid_backbone=backbone, use_ppm=True, num_classes=num_classes, **kwargs)
    model.default_cfg = default_cfgs['vit_small_efficientnet_b6']
    return model

@register_model
def vit_small_efficientnet_b7(pretrained=False, **kwargs):
    pretrained_backbone = kwargs.get('pretrained_backbone', True)
    # Use tf_efficientnet_b7 instead of efficientnet_b7
    backbone = timm.create_model('tf_efficientnet_b7.aa_in1k', 
                               pretrained=pretrained_backbone,
                               features_only=True,
                               out_indices=[4])  # Use [4] for B7
    model = VisionTransformer(
        img_size=456, embed_dim=768, depth=8, num_heads=8, mlp_ratio=3,
        hybrid_backbone=backbone, use_ppm=True, num_classes=num_classes, **kwargs)
    model.default_cfg = default_cfgs['vit_small_efficientnet_b7']
    return model

def load_model(checkpoint_path=None, model_name='vit_small_efficientnet_b7'):
    """Load the trained model using the registered model function"""
    if checkpoint_path is None:
        # Default checkpoint path (update this to match your actual path)
        checkpoint_path = 'C:\\Users\\rupes\\OneDrive\\Desktop\\deployableApplication\\cxr_streamlit_starter\\models\\tbmodelb7.pth'
    
    # Create model using timm's create_model with your registered model
    model = create_model(model_name, pretrained=False)
    
    # Enable gradients for all parameters (as in your original code)
    for param in model.parameters(): 
        param.requires_grad = True

    # Create the same head architecture as in your training code
    fc = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(768, 1000, bias=True)),
        ('BN1', nn.BatchNorm1d(1000)),
        ('dropout1', nn.Dropout(0.7)),
        ('fc2', nn.Linear(1000, 512)),
        ('BN2', nn.BatchNorm1d(512)),
        ('swish1', Swish()),
        ('dropout2', nn.Dropout(0.5)),
        ('fc3', nn.Linear(512, 128)),
        ('BN3', nn.BatchNorm1d(128)),
        ('swish2', Swish()),
        ('fc4', nn.Linear(128, num_classes)),
        ('output', nn.Softmax(dim=1))
    ]))
    
    model.head = fc
    
    # Load checkpoint
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    model.load_state_dict(checkpoint)
            else:
                model.load_state_dict(checkpoint)
            print(f"Model loaded successfully from {checkpoint_path}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Loading model with pretrained backbone...")
            # Fallback to pretrained backbone if checkpoint loading fails
            model = create_model(model_name, pretrained=False, pretrained_backbone=True)
            model.head = fc
    else:
        print(f"Checkpoint not found at {checkpoint_path}")
        print("Loading model with pretrained backbone...")
        model = create_model(model_name, pretrained=False, pretrained_backbone=True)
        model.head = fc
    
    model.to(device)
    model.eval()
    return model

def preprocess_image(image):
    """Preprocess image for model inference"""
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Define transforms (same as used in training)
    test_size = int((256 / 224) * img_size)
    transform = transforms.Compose([
        transforms.Resize(test_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    # Apply transforms
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor.to(device)

def predict_image(model, image):
    """Make prediction on a single image"""
    with torch.no_grad():
        # Preprocess image
        image_tensor = preprocess_image(image)
        
        # Get model output
        outputs = model(image_tensor)
        
        # Get probabilities
        probabilities = torch.softmax(outputs, dim=1)
        
        # Get prediction
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item() * 100
        
        # Class names
        class_names = ['NORMAL', 'TUBERCULOSIS']
        prediction = class_names[predicted_class]
        
        # Return probabilities as list for both classes
        prob_list = probabilities[0].cpu().numpy().tolist()
        
        return prediction, confidence, prob_list

# Test function
def test_model(model_name='vit_small_efficientnet_b0'):
    """Test the model loading and prediction"""
    try:
        model = load_model(model_name=model_name)
        print("Model loaded successfully!")
        
        # Print model parameter count (as in your original code)
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        count = count_parameters(model)
        print(f"The number of parameters of the model is: {count}")
        
        # Test with a dummy image
        dummy_image = Image.new('RGB', (456, 456), color='white')
        prediction, confidence, probabilities = predict_image(model, dummy_image)
        
        print(f"Test prediction: {prediction}")
        print(f"Test confidence: {confidence:.2f}%")
        print(f"Test probabilities: {probabilities}")
        
        return True
    except Exception as e:
        print(f"Test failed: {e}")
        return False

if __name__ == "__main__":
    test_model()