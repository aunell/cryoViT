import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

def recon_overlap_center_modified(cols, rows, patch_dim, total_features, ratio):
    all_rows=[]
    for j in range(0,rows*cols, cols):
        concatenated_arrays = []
        for i in range(cols):
            total_features_ji = total_features[j+i].cpu()
            center_point=patch_dim//2
            left_bound=center_point-patch_dim//(2*ratio)
            right_bound=center_point+patch_dim//(2*ratio)
            total_features_ji = total_features_ji[left_bound:right_bound, left_bound:right_bound, :]
            concatenated_arrays.append(total_features_ji)
        concatenated_array = torch.cat(concatenated_arrays, dim=1)
        all_rows.append(concatenated_array)
    concatenated_array = torch.cat(all_rows, dim=0)
    return concatenated_array

def create_feature_grid(rows,cols, inner_crop_patch_dim, cropped_features, feat_dim):
    reformatted_features = np.zeros((rows*inner_crop_patch_dim, cols*inner_crop_patch_dim, feat_dim)) #linearization of crops into grid of crops
    for i in range(rows):
        for j in range(cols):
            crop = cropped_features[i,j]
            reformatted_features[i*inner_crop_patch_dim:(i+1)*inner_crop_patch_dim, j*inner_crop_patch_dim:(j+1)*inner_crop_patch_dim] = crop
    return reformatted_features

def normalize_features(cropped_features):
    mean = torch.mean(cropped_features, dim=(1,2), keepdim=True)
    #need to normalize over each patch's features and not over each crop features --> all 64 patches in an 8 by 8 innercrop should be 0 mean and 1 std
    #this way, when we compare crops to each other, we are comparing the features of the patches within the crop at the same scale
    std = torch.std(cropped_features, dim=(1,2), keepdim=True)
    cropped_features = (cropped_features - mean) / std
    return cropped_features

def extract_inner_crops(total_features, inner_crop_patch_dim, rows, cols, feat_dim):
    start_idx=total_features.shape[1]//2-inner_crop_patch_dim//2
    end_idx=total_features.shape[1]//2+inner_crop_patch_dim//2
    cropped_features = total_features[:, start_idx:end_idx, start_idx:end_idx, :]
    assert(cropped_features.shape == (rows*cols, inner_crop_patch_dim, inner_crop_patch_dim, feat_dim)) 
    return cropped_features