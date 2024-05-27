import sys
sys.path.append('/pasteur/u/aunell')
from PIL import Image
import matplotlib.pyplot as plt
from cryoViT.features.load_model import return_features, load_model
from cryoViT.features.crop_image import create_crops
from cryoViT.features.dimensionality import reduce_features
from cryoViT.features.visualize import plot_umap
from cryoViT.features.image_recon import create_feature_grid, normalize_features, extract_inner_crops
import argparse
import torch
import numpy as np
import torch.nn as nn

def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="Process some integers.")

    # Add the arguments
    parser.add_argument('--output_dir', type=str, default="/pasteur/u/aunell/cryoViT/features/image_large_unnormalized.png", help='The output directory')
    parser.add_argument('--image', type=str, default="/pasteur/u/aunell/cryoViT/data/sample_data/original/image_test_L25_001_16.png", help='The image file path')
    parser.add_argument('--crop_size', type=int, default=448, help='The crop size') #3 choices [224, 448, 896, 3010, 2800]
    parser.add_argument('--backbone', type=str, default='dinov2_vitg14_reg', help='The backbone model') #4 choices [dinov2_vitg14_reg, dinov2_vitl14_reg, dinov2_vitb14_reg, dinov2_vits14_reg]
    parser.add_argument('--include_hsv', type=str, default="False", help='Include HSV in the plot') #2 choices [True, False]
    parser.add_argument('--stride', type=int, default=112, help='The inner crop size') #3 choices [1, 2, 8]
    parser.add_argument('--load_features', type=str, default="False", help='Load features from file')
    #stride is center crop size
    # Parse the arguments
    args = parser.parse_args()
    img= Image.open(args.image).convert('RGB')
    width, height = img.size  
    crop_size = args.crop_size 
    stride=args.stride #args.crop_size//14//4
    backbone = args.backbone
    inner_crop_patch_dim = stride//14
    print('image size', width, height)

    crops, width_padding_to_remove, height_padding_to_remove, rows, cols = create_crops(img, crop_size, stride=stride) #rows of overlapping crops -> width_pad//stride
    dinov2, feat_dim, patch_h, patch_w = load_model(backbone, crop_size)
    print('model loaded')

    if args.load_features != "False":
        total_features = torch.load(args.load_features)
    else:
        total_features = return_features(crops, dinov2)  #returns patch_len, crop/14xcrop/14, feat_dim
        assert(total_features.shape == (rows*cols, patch_h*patch_w, feat_dim)) 
        total_features = total_features.reshape(rows*cols, patch_h, patch_w, feat_dim).cpu()

    cropped_features = extract_inner_crops(total_features, inner_crop_patch_dim, rows, cols, feat_dim)#([896, 8, 8, 384]) compute mean and std for each 8x8 region's individual patch

    # cropped_features = normalize_features(cropped_features)
    cropped_features = cropped_features.reshape(rows, cols, inner_crop_patch_dim, inner_crop_patch_dim, feat_dim)# (896,8,8,384) --> 32,28, 8, 8, 384
    assert(cropped_features.shape == (rows, cols, inner_crop_patch_dim, inner_crop_patch_dim, feat_dim)) 
    # print(rows,cols)
    # breakpoint()
    reformatted_features = create_feature_grid(rows,cols, inner_crop_patch_dim, cropped_features, feat_dim) # want first dimension of each of the 896 patches to be lined up linearly until you reach cols entrie
    i=args.output_dir.split('.')[0][-1]
    if height_padding_to_remove != 0:
        reformatted_features = reformatted_features[:-height_padding_to_remove, :, :]
    if width_padding_to_remove != 0:
        reformatted_features = reformatted_features[:, :-width_padding_to_remove, :]
    # breakpoint()
    # torch.save(reformatted_features, f'/pasteur/u/aunell/cryoViT/features/features_g_{i}.pt')
    h_final, w_final = reformatted_features.shape[0], reformatted_features.shape[1]
    cropped_features_flat = reduce_features(reformatted_features, feat_dim, n_components=feat_dim//3)
    assert(cropped_features_flat.shape == (h_final*w_final, 3))
    
    plot_umap(cropped_features_flat, img, h_final, w_final, output_dir=args.output_dir, include_hsv=args.include_hsv, i=i)

if __name__ == "__main__":
    main()