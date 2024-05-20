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
    print(f'rows pad: {rows-height//crop_size}, cols pad: {cols-width//crop_size}')
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

def get_overlapping(image, crop_size=448, stride=2):
    """crops image into crop_size by crop_size dimensions
    crop size must be divisible by patch size (14 for dinov2)"""
    width, height = image.size
    padding_size_right = crop_size - (width % crop_size) if width % crop_size != 0 else 0
    padding_size_bottom = crop_size - (height % crop_size) if height % crop_size != 0 else 0
    image = ImageOps.expand(image, border=(0, 0, padding_size_right, padding_size_bottom), fill='black')
    width_pad, height_pad = image.size
    rows = height_pad // crop_size*stride
    cols = width_pad // crop_size*stride
    rows_pad = rows-height_pad // crop_size*stride
    cols_pad = cols-width_pad // crop_size*stride
    patches = []
    for i in range(0, height_pad, crop_size//stride):
        for j in range(0, width_pad, crop_size//stride):
            try:
                patch = F.crop(image, i, j, crop_size, crop_size)
            except:
                print(f'Error at i: {i}, j: {j}')
                continue
            assert (patch.size == (crop_size, crop_size))

            patches.append(patch)

    return patches, width_pad, height_pad, rows, cols, rows_pad, cols_pad

def get_overlapping_center(image, crop_size=448, stride=2):
    """crops image into crop_size by crop_size dimensions
    crop size must be divisible by patch size (14 for dinov2)
    stride is pixel values that we shift to the rigjht or down for each image patch
    where a patch is crop_size//14
    
    stride is ratio of center patch that is kept to crop size ie stride of 2 means we are keeping a 224 x 224 patch (224/14 x 224/14)"""
    width, height = image.size
    inner_crop_dimension = crop_size//stride
    center_image_half_size=inner_crop_dimension//14//2
    width+=center_image_half_size*2
    height+=center_image_half_size*2
    padding_size_right = crop_size - (width % crop_size) if width % crop_size != 0 else 0
    padding_size_bottom = crop_size - (height % crop_size) if height % crop_size != 0 else 0
    image = ImageOps.expand(image, border=(center_image_half_size, center_image_half_size, padding_size_right+center_image_half_size, padding_size_bottom+center_image_half_size), fill='black')
    width_pad, height_pad = image.size
    rows = height_pad // crop_size*stride
    cols = width_pad // crop_size*stride
    rows_pad = rows-height_pad // crop_size*stride
    cols_pad = cols-width_pad // crop_size*stride
    patches = []
    print(inner_crop_dimension)
    for i in range(0, height_pad, inner_crop_dimension):
        for j in range(0, width_pad, inner_crop_dimension):
            patch = F.crop(image, i, j, crop_size, crop_size)
            assert (patch.size == (crop_size, crop_size))

            patches.append(patch)
    print(len(patches))
    print(rows, cols)
    assert(len(patches)==rows*cols)
    return patches, width_pad, height_pad, rows, cols, rows_pad, cols_pad