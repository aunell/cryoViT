import sys
sys.path.append('/pasteur/u/aunell')
from PIL import Image
import matplotlib.pyplot as plt
from cryoViT.features.load_model import return_features, load_model
from cryoViT.features.crop_image import get_cropped, get_overlapping
from cryoViT.features.dimensionality import find_pca, find_umap
from cryoViT.features.visualize import plot_umap
from cryoViT.features.image_recon import recon_patch, recon_overlap
import argparse
import torch
import numpy as np

def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="Process some integers.")

    # Add the arguments
    parser.add_argument('--output_dir', type=str, default="/pasteur/u/aunell/cryoViT/features/", help='The output directory')
    parser.add_argument('--image', type=str, default="/pasteur/u/aunell/cryoViT/data/sample_data/original/image_test_L25_001_16.png", help='The image file path')
    parser.add_argument('--crop_size', type=int, default=224, help='The crop size') #3 choices [224, 448, 896, 3010, 2800]
    parser.add_argument('--dimensionality', type=str, default='Both', help='PCA, UMAP, or Both') #3 choices [PCA, UMAP, Both]
    parser.add_argument('--backbone', type=str, default='dinov2_vitg14_reg', help='The backbone model') #4 choices [dinov2_vitg14_reg, dinov2_vitl14_reg, dinov2_vitb14_reg, dinov2_vits14_reg]
    parser.add_argument('--include_hsv', type=bool, default=True, help='Include HSV in the plot') #2 choices [True, False]

    # Parse the arguments
    args = parser.parse_args()
    img= Image.open(args.image).convert('RGB')
    width, height = img.size   
    # patches, width_pad, height_pad, rows, cols = get_cropped(img, crop_size=args.crop_size)
    patches, width_pad, height_pad, rows, cols = get_overlapping(img, crop_size=args.crop_size)
    dinov2, feat_dim, patch_h, patch_w = load_model(backbone=args.backbone, crop_size=args.crop_size)
    total_features = return_features(patches, dinov2)  #returns patch_len, crop/14xcrop/14, feat_dim, overlaps every crop//2
    total_features = total_features.reshape(rows*cols, patch_h, patch_w, feat_dim).cpu()
    print('total_features shape', total_features.shape) #([168, 32, 32, 1536]) for overlap, [42, 32, 32, 1536]) for regular
    # concatenated_array = recon_patch(patches, cols, rows, patch_h, patch_w, feat_dim, total_features)
    concatenated_array = recon_overlap(patches, cols, rows, patch_h, patch_w, feat_dim, total_features)
    print('concat array shape', concatenated_array.shape) #448, 384, 1536 ([224, 192, 1536]) for crop
    rows, cols =  rows//2, cols//2
    assert(concatenated_array.shape == (rows*patch_h, cols*patch_w, feat_dim))
    if args.dimensionality == 'Both':
            total_features_j = concatenated_array.reshape(patch_h * patch_w*rows*cols, feat_dim)
            assert(total_features_j.shape == (patch_h * patch_w*cols*rows, feat_dim))
            total_features_j = find_pca(total_features_j, n_components=128)
    total_features_j = find_umap(total_features_j, n_components=3)
    assert(total_features_j.shape == (patch_h * patch_w*cols*rows, 3))
    plot_umap(total_features_j, img, patch_h*rows, patch_w*cols, width_pad, height_pad, width, height, output_dir=f"{args.output_dir}/UMAP_overlap_224_hsv.png", include_hsv=args.include_hsv)

if __name__ == "__main__":
    main()