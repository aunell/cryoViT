import sys
sys.path.append('/pasteur/u/aunell')
from PIL import Image
import matplotlib.pyplot as plt
from cryoViT.features.load_model import return_features, load_model
from cryoViT.features.crop_image import get_cropped, get_overlapping, get_overlapping_center
from cryoViT.features.dimensionality import find_pca, find_umap
from cryoViT.features.visualize import plot_umap
from cryoViT.features.image_recon import recon_patch, recon_overlap, recon_overlap_center
import argparse
import torch
import numpy as np

def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="Process some integers.")

    # Add the arguments
    parser.add_argument('--output_dir', type=str, default="/pasteur/u/aunell/cryoViT/features/", help='The output directory')
    parser.add_argument('--image', type=str, default="/pasteur/u/aunell/cryoViT/data/sample_data/original/image_test_L25_001_16.png", help='The image file path')
    parser.add_argument('--crop_size', type=int, default=448, help='The crop size') #3 choices [224, 448, 896, 3010, 2800]
    parser.add_argument('--dimensionality', type=str, default='Both', help='PCA, UMAP, or Both') #3 choices [PCA, UMAP, Both]
    parser.add_argument('--backbone', type=str, default='dinov2_vitg14_reg', help='The backbone model') #4 choices [dinov2_vitg14_reg, dinov2_vitl14_reg, dinov2_vitb14_reg, dinov2_vits14_reg]
    parser.add_argument('--include_hsv', type=str, default="False", help='Include HSV in the plot') #2 choices [True, False]

    # Parse the arguments
    args = parser.parse_args()
    def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)

    set_seed(0)  # or any other number
    img= Image.open(args.image).convert('RGB')
    width, height = img.size   
    stride=2
    print('image size', width, height)
    # patches, width_pad, height_pad, rows, cols = get_cropped(img, crop_size=args.crop_size)
    patches, width_pad, height_pad, rows, cols, rows_pad, cols_pad = get_overlapping(img, crop_size=args.crop_size, stride=stride)
    # patches, width_pad, height_pad, rows, cols= get_overlapping_center(img, crop_size=args.crop_size)
    dinov2, feat_dim, patch_h, patch_w = load_model(backbone=args.backbone, crop_size=args.crop_size)
    # total_features = return_features(patches, dinov2)  #returns patch_len, crop/14xcrop/14, feat_dim, overlaps every crop//2
    # total_features = total_features.reshape(rows*cols, patch_h, patch_w, feat_dim).cpu()
    # torch.save(total_features, '/pasteur/u/aunell/cryoViT/features/features_0518.pt')
    total_features = torch.load('/pasteur/u/aunell/cryoViT/features/features_0518.pt')
    print('total_features shape', total_features.shape) #([168, 32, 32, 1536]) for overlap, [42, 32, 32, 1536]) for regular
    # concatenated_array = recon_patch(patches, cols, rows, patch_h, patch_w, feat_dim, total_features)
    concatenated_array = recon_overlap_center(patches, cols, rows, patch_h, patch_w, feat_dim, total_features)
    # concatenated_array = recon_overlap(patches, cols, rows, patch_h, patch_w, feat_dim, total_features, rows_pad, cols_pad, stride=stride)
    print('concat array shape', concatenated_array.shape) #448, 384, 1536 ([224, 192, 1536]) for crop
    # rows, cols = rows//stride, cols//stride
    patch_h, patch_w = patch_h//stride, patch_w//stride
    # assert(concatenated_array.shape == (rows*patch_h, cols*patch_w, feat_dim))
    # total_features_j = concatenated_array.reshape(patch_h * patch_w*rows*cols, feat_dim)
    total_features_j = concatenated_array.reshape(-1, feat_dim)

    # assert(total_features_j.shape == (patch_h * patch_w*cols*rows, feat_dim))
    if args.dimensionality != 'UMAP':
            total_features_j = find_pca(total_features_j, n_components=512)
    if args.dimensionality!= 'PCA':
         total_features_j = find_umap(total_features_j, n_components=3)
    # assert(total_features_j.shape == (patch_h * patch_w*cols*rows, 3))
    plot_umap(total_features_j, img, patch_h*rows, patch_w*cols, width_pad, height_pad, width, height, output_dir=f"{args.output_dir}/UMAP_TEST.png", include_hsv=args.include_hsv)

if __name__ == "__main__":
    main()