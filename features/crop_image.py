from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor, ToPILImage
from PIL import ImageOps

def get_cropped(image, crop_size=896):
    """crops image into crop_size by crop_size dimensions
    crop size must be divisible by patch size (14 for dinov2)"""
    width, height = image.size
    padding_size_right = crop_size - (width % crop_size) if width % crop_size != 0 else 0
    padding_size_bottom = crop_size - (height % crop_size) if height % crop_size != 0 else 0
    image = ImageOps.expand(image, border=(0, 0, padding_size_right, padding_size_bottom), fill='black')
    width_pad, height_pad = image.size
    rows = height_pad // crop_size
    cols = width_pad // crop_size
    print(f'width_pad: {width_pad}, height_pad: {height_pad}')
    print(f'width: {width}, height: {height}')
    print(f'rows: {rows}, cols: {cols}')
    patches = []
    for i in range(0, height, crop_size):
        for j in range(0, width, crop_size):
            if i+crop_size>height_pad:
                i = height-crop_size
                raise ValueError('i+crop_size>height_pad')
            if j+crop_size>width_pad:
                j = width-crop_size
                raise ValueError('i+crop_size>width_pad')
            patch = F.crop(image, i, j, crop_size, crop_size)
            assert (patch.size == (crop_size, crop_size))

            patches.append(patch)

    return patches, width_pad, height_pad, rows, cols

def get_overlapping(image, crop_size=448):
    """crops image into crop_size by crop_size dimensions
    crop size must be divisible by patch size (14 for dinov2)"""
    width, height = image.size
    padding_size_right = crop_size - (width % crop_size) if width % crop_size != 0 else 0
    padding_size_bottom = crop_size - (height % crop_size) if height % crop_size != 0 else 0
    image = ImageOps.expand(image, border=(0, 0, padding_size_right, padding_size_bottom), fill='black')
    width_pad, height_pad = image.size
    rows = height_pad // crop_size*2
    cols = width_pad // crop_size*2
    print(f'width_pad: {width_pad}, height_pad: {height_pad}')
    print(f'width: {width}, height: {height}')
    print(f'rows: {rows}, cols: {cols}')
    patches = []
    for i in range(0, height_pad, crop_size//2):
        for j in range(0, width_pad, crop_size//2):
            try:
                patch = F.crop(image, i, j, crop_size, crop_size)
            except:
                continue
            assert (patch.size == (crop_size, crop_size))

            patches.append(patch)

    return patches, width_pad, height_pad, rows, cols