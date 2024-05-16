import colorsys
import matplotlib.pyplot as plt

def plot_umap(umap_features, ground_truth, patch_h, patch_w, width_pad, height_pad,
              width, height, output_dir=None, include_hsv=True):
    # Convert RGB to HSV, adjust saturation and value, then convert back to RGB
    if include_hsv:
        for i in range(umap_features.shape[0]):
            hsv = colorsys.rgb_to_hsv(*umap_features[i])
            hsv = (hsv[0], 0.9, 0.9)
            umap_features[i] = colorsys.hsv_to_rgb(*hsv)

    # Create 1x2 subplots
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    row_padding = (height_pad-height)//14
    col_padding = (width_pad-width)//14
    umap_features = umap_features.reshape(patch_h,patch_w, 3)
    umap_features = umap_features[:-row_padding, :-col_padding, :]
    # axs[0].imshow(umap_features.reshape(patch_h, patch_w, 3))
    axs[0].imshow(umap_features)
    axs[0].set_title('UMAP Processed')

    # Plot ground truth
    axs[1].imshow(ground_truth)
    axs[1].set_title('Ground Truth Total')

    plt.savefig(output_dir)
    plt.show()