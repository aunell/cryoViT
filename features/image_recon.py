import torch
import numpy as np
from tqdm import tqdm

def recon_patch(patches, cols, rows, patch_h, patch_w, feat_dim, total_features):
    all_rows=[]
    for j in range(0,len(patches), cols):
        concatenated_arrays = []
        for i in range(cols):
            total_features_ji = total_features[j+i].cpu()
            concatenated_arrays.append(total_features_ji)
        concatenated_array = torch.cat(concatenated_arrays, dim=1)
        all_rows.append(concatenated_array)
    concatenated_array = torch.cat(all_rows, dim=0)
    assert(concatenated_array.shape == (patch_h*rows, patch_w*cols, feat_dim))
    return concatenated_array

def recon_overlap(patches, cols, rows, patch_h, patch_w, feat_dim, total_features,rows_pad, cols_pad, stride=2):
    all_rows=[]
    neighbors = find_neighbors(rows-rows_pad, cols-cols_pad) #([168, 32, 32, 1536]), 14
    for j in tqdm(range(0,len(patches), cols*stride)):
        concatenated_arrays = []
        for i in range(cols):
            total_features_ji = total_features[j+i].cpu()
            print("Before blend_features: ", total_features_ji)
            total_features_ji = blend_features(total_features_ji, j+i, neighbors, total_features, stride)
            total_features[j+i] = total_features_ji
            print("After blend_features: ", total_features_ji)
            if i%stride!=0:
                continue
            concatenated_arrays.append(total_features_ji)
        concatenated_array = torch.cat(concatenated_arrays, dim=1)
        all_rows.append(concatenated_array)
    concatenated_array = torch.cat(all_rows, dim=0)
    rows_reduced = rows  // stride
    cols_reduced = cols  // stride
    assert(concatenated_array.shape == (patch_h*rows_reduced, patch_w*cols_reduced, feat_dim))
    return concatenated_array

def find_neighbors(rows, cols):
    neighbors_upper = {}
    neighbors_lower = {}
    neighbors_left= {}
    neighbors_right = {}
    for i in range(0, rows-1):
        for j in range(cols-1):
            index = i*cols + j

            # Check if there is a patch above
            if i > 0:
                neighbors_upper[index]=(i-1) * cols + j
            # Check if there is a patch below
            if i < rows - 1:
                neighbors_lower[index] = (i+1) * cols + j
            # Check if there is a patch to the left
            if j > 0:
                neighbors_left[index] = i * cols + (j-1)
            # Check if there is a patch to the right
            if j < cols - 1:
               neighbors_right[index] = i * cols + (j+1)

    return [neighbors_upper, neighbors_lower, neighbors_left, neighbors_right]

def blend_features(features_ji, index, neighbors, total_features, stride=2):
    blended_features = features_ji
    neighbors_upper, neighbors_lower, neighbors_left, neighbors_right = neighbors
    neighbors_upper = neighbors_upper.get(index, None)
    neighbors_lower = neighbors_lower.get(index, None)
    neighbors_left = neighbors_left.get(index, None)
    neighbors_right = neighbors_right.get(index, None)
    # print(blended_features.shape) #torch.Size([32, 32, 1536])
    if neighbors_upper is not None:
        blended_features= blend_upper_neighbor(blended_features, total_features[neighbors_upper], stride)
        assert(blended_features.shape == features_ji.shape)
    if neighbors_lower is not None:
        blended_features= blend_lower_neighbor(blended_features, total_features[neighbors_lower], stride)
        assert(blended_features.shape == features_ji.shape)
    if neighbors_left is not None:
        blended_features= blend_left_neighbor(blended_features, total_features[neighbors_left], stride)
        assert(blended_features.shape == features_ji.shape)
    if neighbors_right is not None:
        blended_features= blend_right_neighbor(blended_features, total_features[neighbors_right], stride)
        assert(blended_features.shape == features_ji.shape)
    return blended_features

##B IS THE NEIGHBOR, A IS ORIGINAL
def blend_upper_neighbor(top, bottom,stride):
    half_val = top.shape[0] // stride
    top_half_A = top[:-half_val, :, :]
    bottom_half_A = top[-half_val:, :, :]
    bottom_half_B = bottom[half_val:, :, :]

    average_top_half = top_half_A +  bottom_half_B / 2

    new_array = torch.cat([average_top_half, bottom_half_A], axis=0)
    return new_array

def blend_lower_neighbor(top, bottom, stride):
    half_val = top.shape[0] // stride
    top_half_A = top[:half_val, :, :]
    bottom_half_A = top[half_val:, :, :]
    top_half_B = bottom[:-half_val, :, :]
    average_bottom_half = (bottom_half_A +  top_half_B) / 2
    new_array = torch.cat([top_half_A, average_bottom_half], axis=0)
    return new_array

def blend_left_neighbor(left, right, stride):
    half_val = left.shape[1] // stride
    left_half_A = left[:, :-half_val, :]
    right_half_A = left[:, -half_val:, :]
    right_half_B = right[:, half_val:, :]

    average_left_half = (left_half_A + right_half_B) / 2

    new_array = torch.cat([average_left_half, right_half_A], axis=1)
    return new_array

def blend_right_neighbor(left, right, stride):
    half_val = left.shape[1] // stride
    left_half_A = left[:, :half_val, :]
    right_half_A = left[:, half_val:, :]
    left_half_B = right[:, :-half_val, :]
    average_right_half = (right_half_A + left_half_B) / 2

    return_val = torch.cat([left_half_A, average_right_half], axis=1)
    return return_val