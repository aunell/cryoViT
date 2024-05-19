# large model
import torch
from io import BytesIO
import requests
from PIL import Image
from torchvision import transforms
import numpy as np
import os
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

# torch.hub.set_dir('/pasteur/u/aunell/dino')
def load_model(backbone='dinov2_vitg14_reg', crop_size=896):
    """Loads a model from the Facebook Research DINO repository.
    crop_size must be divisible by patch size (14 for dinov2)
    """
    set_seed(0) 
    model_path = f'/pasteur/u/aunell/cryoViT/model/{backbone}.pth'
    if os.path.exists(model_path):
        print("Model file exists.")
        dinov2 = torch.hub.load('facebookresearch/dinov2', backbone).cuda()
        dinov2.load_state_dict(torch.load(model_path))
    else:
        dinov2 = torch.hub.load('facebookresearch/dinov2', backbone).cuda()
        # Save the model to disk
        torch.save(dinov2.state_dict(), model_path)
    patch_size = dinov2.patch_size  # patchsize=14
    patch_h = crop_size // patch_size
    patch_w = crop_size // patch_size
    feat_dim = 1536  # vitg14
    if backbone == 'dinov2_vits14_reg':
        feat_dim = 384
    elif backbone == 'dinov2_vitb14_reg':
        feat_dim = 768
    elif backbone == 'dinov2_vitl14_reg':    
        feat_dim = 1024
    elif backbone == 'dinov2_vitg14_reg':
        feat_dim = 1536
    else:
        raise ValueError('backbone not supported')
    return dinov2, feat_dim, patch_h, patch_w

def return_features(patches, model):
  """
  Returns features from an image using a ViT."""
  total_features  = []
  transform = transforms.Compose([              
                                transforms.ToTensor(),                    
                                transforms.Normalize(                      
                                mean=[0.485, 0.456, 0.406],                
                                std=[0.229, 0.224, 0.225]              
                                )])
  for i, img in enumerate(patches):
    with torch.no_grad():
        img_t = transform(img).cuda()
        features_dict = model.forward_features(img_t.unsqueeze(0))
        features = features_dict['x_norm_patchtokens']
        total_features.append(features)
  total_features = torch.cat(total_features, dim=0) #expected torch.Size([1, 21904, 1536])
  assert(total_features.size()[0] == len(patches))
  return total_features 

def return_features_overlapped(patches, model, cols, rows):
    """
    Returns features from an image using a ViT."""
    set_seed(0) 
    total_features  = []
    transform = transforms.Compose([              
                                    transforms.ToTensor(),                    
                                    transforms.Normalize(                      
                                    mean=[0.485, 0.456, 0.406],                
                                    std=[0.229, 0.224, 0.225]              
                                    )])
    for j in range(rows):
        for i in range(cols):
            with torch.no_grad():
                if i!=0:
                    img_l = patches[j*i-1]
                if i!=cols-1:
                    img_r = patches[j*i+1]
                if j!=0:
                    img_u = patches[i*j-1]
                if j!=rows-1:
                    img_d = patches[i*j+1]
                
                img= patches[i]
                img_t = transform(img).cuda()
                features_dict = model.forward_features(img_t.unsqueeze(0))
                features = features_dict['x_norm_patchtokens']
                total_features.append(features)
        total_features = torch.cat(total_features, dim=0) #expected torch.Size([1, 21904, 1536])
        assert(total_features.size()[0] == len(patches))
    return total_features