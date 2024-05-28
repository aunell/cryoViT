import torch
import numpy as np
import sys
sys.path.append('/pasteur/u/aunell')
from cryoViT.data.preprocessing import load_text, draw_circle_on_image, create_trust_region_coords, draw_square_on_image
from typing import List, Tuple, Iterator, Dict
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
from cryoViT.features.dimensionality import reduce_features
from cryoViT.features.visualize import color_features
from PIL import ImageDraw
import pickle


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
                    radius = int(rows[key]['diameter']) // 14
                    vesicle_information = ((center_x, center_y), radius)
                    vesicle_locations.append(vesicle_information)
                except Exception as e:
                    print("Error is", e, ": Invalid text parsing")
                    print(rows[key])
        return vesicle_locations

    def extract_features(self, vesicle_locations: List[Tuple[Tuple[int,int], int]]) -> dict[int: List[np.array]]:
        vesicle_features = {}
        for center, radius in vesicle_locations:
            i = center[0] - radius
            j = center[1] - radius
            if radius not in vesicle_features:
                vesicle_features[radius] = []
            vesicle_features[radius].append(self.img_features[j:j+2*radius, i:i+2*radius, :])
        return vesicle_features # 32,4,4,1536
    
    def average_similarity_vesicle(self, vesicle_locations: List[Tuple[Tuple[int, int], int]]) -> Dict[int, Tuple[float, np.array]]:
        similarity_dictionary={}
        vesicle_features= self.extract_features(vesicle_locations)
        for radius in vesicle_features.keys():
            avg_vesicle = np.mean(vesicle_features[radius], axis=0)
            total_similarity = 0
            for i in range(len(vesicle_features[radius])):
                similarity = cosine_similarity(vesicle_features[radius][i].reshape(1,-1), avg_vesicle.reshape(1,-1))
                total_similarity += similarity
            avg_similarity = total_similarity / len(vesicle_features[radius])
            avg_vesicle = np.array(avg_vesicle).astype(np.uint8)
            similarity_dictionary[radius] = avg_similarity, avg_vesicle
        return similarity_dictionary

    def sliding_window(self, tensor: torch.Tensor, window_size: int, stride: int) -> Iterator[torch.Tensor]:
        # Create a sliding window over the tensor
        for i in range(0, tensor.shape[0] - window_size + 1, stride):
            for j in range(0, tensor.shape[1] - window_size + 1, stride):
                yield i,j

    def compute_similarities(self, total_features: torch.Tensor, avgsimilaritydict: Dict[int, Tuple[float, np.array]], stride: int, percent_vesicles_selected: float) -> Dict[Tuple, float]:
        potential_vesicles = {} #(i,j), radius: similarity
        min_avg_similarity=np.inf
        for radius, (avg_similarity, avg_vesicle) in avgsimilaritydict.items():
            diameter = 2*radius
            for window_coord in self.sliding_window(total_features, diameter, stride):
                # Flatten the window and avg_vesicle tensors before computing cosine similarity
                i,j = window_coord
                window = total_features[i:i+diameter, j:j+diameter, :]
                similarity = cosine_similarity(window.reshape(1,-1), avg_vesicle.reshape(1,-1))
                potential_vesicles[(window_coord, radius)] = similarity
            if avg_similarity < min_avg_similarity:
                min_avg_similarity = avg_similarity
        # return potential_vesicles
                
        def top_vesicle_choices(label_to_number: dict, percent_vesicles_selected: float, similarity_thresh) -> list:
            sorted_dict = sorted(label_to_number.items(), key=lambda item: item[1], reverse=True)
            top_entries_count = int(len(sorted_dict) * percent_vesicles_selected)
            top_entries = sorted_dict[:top_entries_count]
            top_labels = [entry[0] for entry in top_entries if entry[1].item() > similarity_thresh]
            return top_labels
        print('min avg similarity', min_avg_similarity.item())
        return top_vesicle_choices(potential_vesicles, percent_vesicles_selected, min_avg_similarity.item()/10)    
    
    
    def create_mask_visual(self, img, unmarked_vesicle_locations: List[Tuple[Tuple[int], int]], vesicle_locations: list) -> Image:
        # img = Image.open('/pasteur/u/aunell/cryoViT/data/sample_data/original/image_test_L25_004_16.png').convert('RGB')
        new_width = img.width #// 14
        new_height = img.height #// 14
        img = img.resize((new_width, new_height))
        # img = Image.new('RGB', (new_width, new_height))
        coord_list_to_ignore=[]
        trust_region_coords= create_trust_region_coords(vesicle_locations, padding=10*14)
        for coords, radius in unmarked_vesicle_locations:
            coords = (coords[0]*14, coords[1]*14)
            radius = radius*14
            result=self.create_coords_ignore(coords, radius*4, trust_region_coords)
            coord_list_to_ignore.extend(result)
        for center, radius in vesicle_locations:
            center = (center[0]*14, center[1]*14)
            radius = radius*14
            img = draw_circle_on_image(img, center, radius, color='white')
        for coords in coord_list_to_ignore:
            draw=ImageDraw.Draw(img)
            draw.point(coords, fill="red")
        return img
    
    def create_mask(self, img, unmarked_vesicle_locations: List[Tuple[Tuple[int], int]], vesicle_locations: list, output_dir) -> Image:
        # img = Image.open('/pasteur/u/aunell/cryoViT/data/sample_data/original/image_test_L25_004_16.png').convert('RGB')
        new_width = img.width #// 14
        new_height = img.height # // 14
        img = img.resize((new_width, new_height))
        img = Image.new('RGB', (new_width, new_height))
        coord_list_to_ignore=[]
        trust_region_coords= create_trust_region_coords(vesicle_locations, padding=10*14)
        for coords, radius in unmarked_vesicle_locations:
            coords = (coords[0]*14, coords[1]*14)
            radius = radius*14
            result=self.create_coords_ignore(coords, radius*4, trust_region_coords)
            coord_list_to_ignore.extend(result)
        for center, radius in vesicle_locations:
            center = (center[0]*14, center[1]*14)
            radius = radius*14
            img = draw_circle_on_image(img, center, radius, color='white')
        with open(output_dir, 'wb') as f:
            pickle.dump(coord_list_to_ignore, f)
        return img
    
    def create_coords_ignore(self, coords, radius, trust_region_coords):
        coord_list_to_ignore=[]
        y_coord, x_coord = coords
        trust_region_coords = set(trust_region_coords)
        for i in range(x_coord-radius, x_coord+radius):
            for j in range(y_coord-radius, y_coord+radius):
                if (i,j) not in trust_region_coords:
                    coord_list_to_ignore.append((i,j))
        return coord_list_to_ignore
    
def visualize_images(mask, ground_truth, output_dir):
    ground_truth = ground_truth.resize((mask.width, mask.height))
    img = Image.new("RGB", (mask.width + ground_truth.width, mask.height))
    img.paste(mask, (0, 0))
    img.paste(ground_truth, (mask.width, 0))
    img.save(output_dir)

if __name__ == "__main__":
    for i in range(1,8):
        img_features = torch.load(f'/pasteur/u/aunell/cryoViT/data/training/features/features_g_{i}.pt') # rows,cols, inner_crop_patch_dim, inner_crop_patch_dim, feat_dim
        img_features_color = torch.load(f'/pasteur/u/aunell/cryoViT/classifier/UMAP_features_{i}.pt')
        vec_img = color_features(img_features_color, i)
        # vec_img =Image.open(f'/pasteur/u/aunell/cryoViT/classifier/mask_7.png').convert('RGB')
        img = Image.open(f'/pasteur/u/aunell/cryoViT/data/sample_data/original/image_test_L25_00{i}_16.png').convert('RGB')
        text= f'/pasteur/u/aunell/cryoViT/data/sample_data/l25/L25_00{i}_16.txt'
        stride = 1
        mask_creator = MaskCreator(img_features, text)
        vesicle_locations = mask_creator.get_vesicle_locations() #[(center, radius), (center, radius), ...]
        avg_similarity_dict = mask_creator.average_similarity_vesicle(vesicle_locations)
        unmarked_vesicle_locations = mask_creator.compute_similarities(img_features, avg_similarity_dict, stride, percent_vesicles_selected=.03)
        mask_visual = mask_creator.create_mask_visual(vec_img, unmarked_vesicle_locations, vesicle_locations)
        mask_visual.save(f'/pasteur/u/aunell/cryoViT/classifier/mask_visual_{i}.png')
        mask = mask_creator.create_mask(img, unmarked_vesicle_locations, vesicle_locations, f'/pasteur/u/aunell/cryoViT/data/training/ignore/ignore_mask_{i}.pkl')
        mask.save(f'/pasteur/u/aunell/cryoViT/data/training/mask/mask_{i}.png')
        output_dir = f'/pasteur/u/aunell/cryoViT/classifier/mask_compare_{i}.png'
        visualize_images(mask, vec_img, output_dir)
