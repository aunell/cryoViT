import sys
sys.path.append('/pasteur/u/aunell')
from PIL import Image
import matplotlib.pyplot as plt
from cryoViT.features.load_model import return_features, load_model
from cryoViT.features.crop_image import get_cropped
from cryoViT.features.dimensionality import find_pca, find_umap
from cryoViT.features.visualize import plot_umap
import argparse
import torch
import numpy as np

def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="Process some integers.")

    # Add the arguments
    parser.add_argument('--output_dir', type=str, required=True, help='The output directory')
    parser.add_argument('--image', type=str, required=True, help='The image file path')
    parser.add_argument('--crop_size', type=int, default=448, help='The crop size') #3 choices [224, 448, 896]
    parser.add_argument('--dimensionality', type=str, default='Both', help='PCA, UMAP, or Both') #3 choices [PCA, UMAP, Both]
    parser.add_argument('--backbone', type=str, default='dinov2_vitg14_reg', help='The backbone model') #4 choices [dinov2_vitg14_reg, dinov2_vitl14_reg, dinov2_vitb14_reg, dinov2_vits14_reg]
    parser.add_argument('--include_hsv', type=bool, default=True, help='Include HSV in the plot') #2 choices [True, False]

    # Parse the arguments
    args = parser.parse_args()
    img= Image.open(args.image).convert('RGB')
    width, height = img.size   
    patches, width_pad, height_pad, rows, cols = get_cropped(img, crop_size=args.crop_size)
    dinov2, feat_dim, patch_h, patch_w = load_model(backbone=args.backbone, crop_size=args.crop_size)
    total_features = return_features(patches, dinov2) 
    #torch.Size([12, 4096, 1536])
    # total_features= total_features.reshape(total_features.shape[0]*total_features.shape[1], feat_dim) #torch.Size([6144, 256])
    total_features = total_features.reshape(rows*cols, patch_h, patch_w, feat_dim).cpu()
    all_rows=[]
    for j in range(0,len(patches), cols):
        concatenated_arrays = []
        for i in range(cols):
            total_features_ji = total_features[j+i].cpu()
            concatenated_arrays.append(total_features_ji)
        concatenated_array = torch.cat(concatenated_arrays, dim=1)
        all_rows.append(concatenated_array)
    concatenated_array = torch.cat(all_rows, dim=0)
    #TODO NORMALIZE
    assert(concatenated_array.shape == (rows*patch_h, cols*patch_w, feat_dim))
    if args.dimensionality == 'Both':
            total_features_j = concatenated_array.reshape(patch_h * patch_w*rows*cols, feat_dim)
            assert(total_features_j.shape == (patch_h * patch_w*cols*rows, feat_dim))
            total_features_j = find_pca(total_features_j, n_components=128)
    total_features_j = find_umap(total_features_j, n_components=3)
    assert(total_features_j.shape == (patch_h * patch_w*cols*rows, 3))
    plot_umap(total_features_j, img, patch_h*rows, patch_w*cols, width_pad, height_pad, width, height, output_dir=f"{args.output_dir}/UMAP_example_448_{j}.png", include_hsv=args.include_hsv)

if __name__ == "__main__":
    main()