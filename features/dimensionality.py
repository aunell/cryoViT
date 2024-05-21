from sklearn.decomposition import PCA
from umap.umap_ import UMAP
import numpy as np
import torch
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def find_pca(total_features,  n_components=128):
  set_seed(0)
  pca = PCA(n_components=n_components)
  # total_features = total_features.reshape(patch_h * patch_w, feat_dim).cpu()
  pca.fit(total_features)
  print("Total variance captured by PCA: ", np.sum(pca.explained_variance_ratio_))  
  pca_features = pca.transform(total_features)
  # pca_features =(pca_features - pca_features.min(0)) / (pca_features.max(0) - pca_features.min(0))
  assert(pca_features.shape == (total_features.shape[0], n_components))
  return pca_features #(1024, 128)

def find_umap(total_features, n_components=3):
  set_seed(0) 
  umap = UMAP(n_components=n_components)
  umap.fit(total_features)
  umap_features = umap.transform(total_features)
  umap_features_norm = (umap_features - umap_features.min(0)) / (umap_features.max(0) - umap_features.min(0))
  assert(umap_features_norm.shape == (total_features.shape[0], n_components))
  return umap_features_norm #(1024, 3)

