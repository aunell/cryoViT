import sys
sys.path.append('/pasteur/u/aunell')
from PIL import Image
import matplotlib.pyplot as plt
from cryoViT.features.load_model import return_features, load_model
from cryoViT.features.crop_image import get_cropped, get_overlapping, get_overlapping_center
from cryoViT.features.dimensionality import find_pca, find_umap
from cryoViT.features.visualize import plot_umap
from cryoViT.features.image_recon import recon_patch, recon_overlap, recon_overlap_center_modified
import argparse
import torch
import numpy as np
import torch.nn as nn

def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="Process some integers.")

    # Add the arguments
    parser.add_argument('--output_dir', type=str, default="/pasteur/u/aunell/cryoViT/features/", help='The output directory')
    parser.add_argument('--image', type=str, default="/pasteur/u/aunell/cryoViT/data/sample_data/original/image_test_L25_001_16.png", help='The image file path')
    parser.add_argument('--crop_size', type=int, default=448, help='The crop size') #3 choices [224, 448, 896, 3010, 2800]
    parser.add_argument('--dimensionality', type=str, default='Both', help='PCA, UMAP, or Both') #3 choices [PCA, UMAP, Both]
    parser.add_argument('--backbone', type=str, default='dinov2_vits14_reg', help='The backbone model') #4 choices [dinov2_vitg14_reg, dinov2_vitl14_reg, dinov2_vitb14_reg, dinov2_vits14_reg]
    parser.add_argument('--include_hsv', type=str, default="False", help='Include HSV in the plot') #2 choices [True, False]
    parser.add_argument('--ratio', type=int, default=4, help='The ratio size of overlap between two crops, larger number is larger overlap') #3 choices [1, 2, 8]

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
    crop_size = args.crop_size 
    ratio=args.ratio #args.crop_size//14//4
    backbone = args.backbone
    assert(crop_size%(14*ratio)==0)
    print('image size', width, height)
    crops, width_pad, height_pad, rows, cols = get_overlapping_center(img, crop_size, ratio=ratio)
    dinov2, feat_dim, patch_h, patch_w = load_model(backbone, crop_size)
    print('model loaded')
    total_features = return_features(crops, dinov2)  #returns patch_len, crop/14xcrop/14, feat_dim, overlaps every crop//2
    print('total features loaded', total_features.shape) #([168, 32, 32, 1536]) for overlap, [42, 32, 32, 1536]) for regular
    total_features = total_features.reshape(rows*cols, patch_h, patch_w, feat_dim).cpu() #each crop is now patch_hxpatch_w and there are row*cols crops as asserted in get_cropped
    # torch.save(total_features, '/pasteur/u/aunell/cryoViT/features/features_0518.pt')
    # total_features = torch.load('/pasteur/u/aunell/cryoViT/features/features_0518.pt')
    print('total_features shape', total_features.shape) #([168, 32, 32, 1536]) for overlap, [42, 32, 32, 1536]) for regular
    # cropped_features = recon_overlap_center_modified(cols, rows, patch_h, total_features, ratio)
    cropped_features = total_features[:, 12:20, 12:20, :]
    print('concat array shape', cropped_features.shape) #448, 384, 1536 ([224, 192, 1536]) for crop
    inner_crop_h, inner_crop_w = patch_h//ratio, patch_w//ratio
    print('inner_crop_h, inner_crop_w', inner_crop_h, inner_crop_w)
    print('rows, cols', rows, cols)
    # assert(cropped_features.shape == (rows*inner_crop_w, cols*inner_crop_w, feat_dim))
    # print('cropped_features shape', cropped_features.shape)
    # cropped_features_flat = cropped_features.reshape(inner_crop_h*inner_crop_w*rows*cols, feat_dim)
    # print('cropped_features_flat shape', cropped_features_flat.shape)
    #([896, 8, 8, 384]) compute mean and std for each 8x8 region

    mean = torch.mean(cropped_features, dim=(1,2), keepdim=True)
    #need to normalize over each patch's features and not crop features
    std = torch.std(cropped_features, dim=(1,2), keepdim=True)
    cropped_features = (cropped_features - mean) / std
    print('mean,,std,cf', mean.shape, std.shape, cropped_features.shape)
    cropped_features = cropped_features.reshape(rows,cols, inner_crop_h, inner_crop_w, feat_dim)
    # (896,8,8,384) --> 32,28, 8, 8, 384
    print('cropped_features shape', cropped_features.shape) #linearization of crops into grid of crops
    reformatted_array = np.zeros((rows*inner_crop_h, cols*inner_crop_w, feat_dim))
    for i in range(rows):
        for j in range(cols):
            crop = cropped_features[i,j]
            reformatted_array[i*inner_crop_h:(i+1)*inner_crop_h, j*inner_crop_w:(j+1)*inner_crop_w] = crop

    # want first dimension of each of the 896 patches to be lined up linearly until you reach cols entrie
    print('reformatted_array shape', reformatted_array.shape) #448, 384, 1536
    cropped_features=reformatted_array
    #make new empty grid of size dim1xdim2
    #do normalization for each crop separately, normalize over (8,8,dim)
    cropped_features_flat = cropped_features.reshape(-1, feat_dim)
    if args.dimensionality != 'UMAP':
            cropped_features_flat = find_pca(cropped_features_flat, n_components=128)
    if args.dimensionality!= 'PCA':
         cropped_features_flat = find_umap(cropped_features_flat, n_components=3)
    assert(cropped_features_flat.shape == (inner_crop_h*inner_crop_w*cols*rows, 3))
    width_padding_to_remove = 0 #(width_pad-width)//14
    height_padding_to_remove = 0 #(height_pad-height)//14
    plot_umap(cropped_features_flat, img, rows*inner_crop_h, cols*inner_crop_w, width_padding_to_remove, 
              height_padding_to_remove, output_dir=f"{args.output_dir}/UMAP_TEST_2.png", include_hsv=args.include_hsv)

if __name__ == "__main__":
    main()