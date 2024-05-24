from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor, ToPILImage
from PIL import ImageOps

def create_crops(image, crop_size, stride):
    """crops image into crop_size by crop_size dimensions
    crop size must be divisible by patch size (14 for dinov2)
    stride is pixel values that we shift to the rigjht or down for each image patch
    where a patch is crop_size//14
    
    stride is ratio of center patch that is kept to crop size ie stride of 2 means we are keeping a 224 x 224 patch (224/14 x 224/14)"""
    width_og, height_og = image.size
    inner_crop_dimension = crop_size//stride
    inner_crop_dimension = stride
    center_image_padding=(crop_size-stride)//2
    width=width_og+center_image_padding*2
    height=height_og+center_image_padding*2
    padding_size_right = crop_size - (width % crop_size) if width % crop_size != 0 else 0
    padding_size_bottom = crop_size - (height % crop_size) if height % crop_size != 0 else 0
    image = ImageOps.expand(image, border=(center_image_padding, center_image_padding, padding_size_right+center_image_padding, padding_size_bottom+center_image_padding), fill='black')
    width_pad, height_pad = image.size
    rows = (height_pad // crop_size)*(crop_size//inner_crop_dimension)
    cols = (width_pad // crop_size)*(crop_size//inner_crop_dimension)
    crops = []
    for i in range(0, height_pad, inner_crop_dimension):
        for j in range(0, width_pad, inner_crop_dimension):
            cropped_section = F.crop(image, i, j, crop_size, crop_size)
            assert (cropped_section.size == (crop_size, crop_size))

            crops.append(cropped_section)
    assert(len(crops)==rows*cols)
    width_padding_to_remove = (width_pad-width_og)//14
    height_padding_to_remove = (height_pad-height_og)//14
    return crops, width_padding_to_remove, height_padding_to_remove, rows, cols