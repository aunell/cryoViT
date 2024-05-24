import torch
import numpy as np
import sys
sys.path.append('/pasteur/u/aunell')
from cryoViT.data.preprocessing import load_text, draw_circle_on_image
from typing import List, Tuple, Iterator
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
from cryoViT.features.dimensionality import reduce_features
from cryoViT.features.visualize import color_features


class MaskCreator:
    def __init__(self, img_features: np.array , text: str) -> None:
        self.img_features = img_features
        self.text = text

    def get_vesicle_locations(self) -> List[Tuple[Tuple[int,int], int]]:
        rows, circles = load_text(self.text)
        vesicle_locations = []
        for key in rows.keys():
            if key in circles:
                try:
                    center = rows[key]['coords']
                    center_x, center_y = center[0]//14, center[1]//14
                    radius = rows[key]['diameter'] // 14
                    vesicle_locations.append((center_x, center_y))
                except Exception as e:
                    print("Error is", e, ": Invalid text parsing")
                    print(rows[key])
        return vesicle_locations

    def extract_features(self, vesicle_locations: List[Tuple[Tuple[int,int], int]], radius: int) -> List[np.array]:
        vesicle_features = []
        for center in vesicle_locations:
            # hardcoding radius to enable avg vesicle calculation
            i = center[0] - radius
            j = center[1] - radius
            vesicle_features.append(self.img_features[j:j+2*radius, i:i+2*radius, :])
        return vesicle_features # 32,4,4,1536
    
    def min_cosine_similarity(self, vesicle_locations: List[Tuple[int, int]], radius: int) -> float:
        vesicle_features= self.extract_features(vesicle_locations, radius)
        min_similarity = float('inf')
        avg_vesicle = np.mean(vesicle_features, axis=0)
        for i in range(len(vesicle_features)):
            similarity = cosine_similarity(vesicle_features[i].reshape(1,-1), avg_vesicle.reshape(1,-1))
            min_similarity = min(min_similarity, similarity.item())
        avg_vesicle = np.array(avg_vesicle).astype(np.uint8)
        # img = Image.fromarray(avg_vesicle)
        # img.save(f'/pasteur/u/aunell/cryoViT/classifier/potential_vesicle_{i}.png')
        print('min similarity', min_similarity)
        return min_similarity, avg_vesicle

    def sliding_window(self, tensor: torch.Tensor, window_size: int, stride: int) -> Iterator[torch.Tensor]:
        # Create a sliding window over the tensor
        for i in range(0, tensor.shape[0] - window_size + 1, stride):
            for j in range(0, tensor.shape[1] - window_size + 1, stride):
                yield i, j, window_size

    def compute_similarities(self, total_features: torch.Tensor, avg_vesicle: torch.Tensor, 
                             window_size: int, stride: int, min_cosine_similarity: float, threshold_delta: float) -> List:
        potential_vesicles = []
        threshold=min_cosine_similarity+threshold_delta
        for window_coord in self.sliding_window(total_features, window_size, stride):
            # Flatten the window and avg_vesicle tensors before computing cosine similarity
            i, j, window_size = window_coord
            window = total_features[i:i+window_size, j:j+window_size, :]
            similarity = cosine_similarity(window.reshape(1,-1), avg_vesicle.reshape(1,-1))
            if similarity > threshold:
                potential_vesicles.append([i,j,window_size])
        return potential_vesicles
    
    def create_mask(self, img, unmarked_vesicle_locations: list, vesicle_locations: list, radius:int) -> Image:
        # img = Image.open('/pasteur/u/aunell/cryoViT/data/sample_data/original/image_test_L25_004_16.png').convert('RGB')
        new_width = img.width // 14
        new_height = img.height // 14
        img = img.resize((new_width, new_height))
        img = Image.new('RGB', (new_width, new_height))
        if len(unmarked_vesicle_locations) == 0:
            print('NO VESICLES DETECTED')
        else:
            for i,j,window_size in unmarked_vesicle_locations:
                center = (j+radius, i+radius)
                img = draw_circle_on_image(img, center, radius, color='white')
        for center in vesicle_locations:
            img = draw_circle_on_image(img, center, radius, color='green')
        return img

if __name__ == "__main__":
    i=4
    img_features = torch.load(f'/pasteur/u/aunell/cryoViT/features/features_g_{i}.pt') # rows,cols, inner_crop_patch_dim, inner_crop_patch_dim, feat_dim
    img_features_color = torch.load(f'/pasteur/u/aunell/cryoViT/classifier/UMAP_features_{i}.pt')
    vec_img = color_features(img_features_color)
    img = Image.open(f'/pasteur/u/aunell/cryoViT/data/sample_data/original/image_test_L25_00{i}_16.png').convert('RGB')
    text= f'/pasteur/u/aunell/cryoViT/data/sample_data/l25/L25_00{i}_16.txt'
    window_size = 10
    stride = 1
    threshold_delta = -.202 #-0.176
    mask_creator = MaskCreator(img_features, text)
    vesicle_locations = mask_creator.get_vesicle_locations() #[(center, radius), (center, radius), ...]
    min_cosine_similarity, avg_vesicle = mask_creator.min_cosine_similarity(vesicle_locations, window_size//2)
    unmarked_vesicle_locations = mask_creator.compute_similarities(img_features, avg_vesicle, window_size, stride, 
                                                                   min_cosine_similarity, threshold_delta)
    
    mask = mask_creator.create_mask(vec_img, unmarked_vesicle_locations, vesicle_locations, window_size//2)
    print('mask shape', mask.size)
    mask.save(f'/pasteur/u/aunell/cryoViT/classifier/mask_black{i}.png')
