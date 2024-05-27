import colorsys
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageFilter
from PIL.ImageEnhance import Color

def plot_umap(umap_features, ground_truth, patch_h, patch_w, output_dir=None, include_hsv="True", i=1):
    # Convert RGB to HSV, adjust saturation and value, then convert back to RGB
    if include_hsv=="True":
        for i in range(umap_features.shape[0]):
            hsv = colorsys.rgb_to_hsv(*umap_features[i])
            hsv = (hsv[0], 0.9, 0.9)
            umap_features[i] = colorsys.hsv_to_rgb(*hsv)
    umap_features = umap_features.reshape(patch_h,patch_w, 3)
    vec_img = color_features(umap_features, i)

    img = Image.new("RGB", (vec_img.width + ground_truth.width, vec_img.height))
    img.paste(vec_img, (0, 0))
    img.paste(ground_truth, (vec_img.width, 0))
    # img.save(output_dir)

def color_features(features, i) -> Image:
    # per channel normalization and outlier clipping
    features = features - features.mean((0, 1), keepdims=True)
    std = features.std((0, 1), keepdims=True)
    features = features.clip(-2 * std, 2 * std)

    # min max scaling to generate valid RGB tuples
    features = features - features.min((0, 1), keepdims=True)
    features = features / features.max((0, 1), keepdims=True)
    features = (255 * features).astype(np.uint8)
    torch.save(features, f'/pasteur/u/aunell/cryoViT/classifier/UMAP_features_{i}.pt')
    img = Image.fromarray(features)
    converter = Color(img)
    img = converter.enhance(1.25)  # boost saturation
    img = img.resize(
        (img.width * 14, img.height * 14),
        resample=Image.NEAREST,
    ) 
    return img