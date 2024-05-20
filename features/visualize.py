import colorsys
import matplotlib.pyplot as plt
import numpy as np
import torch
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
def plot_umap(umap_features, ground_truth, patch_h, patch_w, width_pad, height_pad,
              width, height, output_dir=None, include_hsv="True", crop_size=448):
    # Convert RGB to HSV, adjust saturation and value, then convert back to RGB
    set_seed(0)
    if include_hsv=="True":
        for i in range(umap_features.shape[0]):
            hsv = colorsys.rgb_to_hsv(*umap_features[i])
            hsv = (hsv[0], 0.9, 0.9)
            umap_features[i] = colorsys.hsv_to_rgb(*hsv)
    # Create 1x2 subplots
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    row_padding = (height_pad-height)//14
    col_padding = (width_pad-width)//14
    print(umap_features.shape)
    print(patch_h, patch_w, 3)
    umap_features = umap_features.reshape(patch_h,patch_w, 3)
    print(row_padding, col_padding)
    if row_padding != 0:
        umap_features = umap_features[:-row_padding, :, :]
    if col_padding != 0:
        umap_features = umap_features[:, :-col_padding, :]
    axs[0].imshow(umap_features)
    axs[0].set_title('UMAP Processed')

    # Plot ground truth
    axs[1].imshow(ground_truth)
    axs[1].set_title('Ground Truth Total')

    plt.savefig(output_dir)
    plt.show()