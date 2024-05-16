from sklearn.decomposition import PCA
from umap.umap_ import UMAP

def find_pca(total_features,  n_components=128):
  pca = PCA(n_components=n_components)
  print('before', total_features.shape)
  # total_features = total_features.reshape(patch_h * patch_w, feat_dim).cpu()
  pca.fit(total_features.cpu())
  pca_features = pca.transform(total_features.cpu())
  assert(pca_features.shape == (total_features.shape[0], n_components))
  return pca_features #(1024, 128)

def find_umap(total_features, n_components=3):
  umap = UMAP(n_components=n_components)
  umap.fit(total_features)
  umap_features = umap.transform(total_features)
  umap_features_norm = (umap_features - umap_features.min(0)) / (umap_features.max(0) - umap_features.min(0))
  assert(umap_features_norm.shape == (total_features.shape[0], n_components))
  return umap_features_norm #(1024, 3)

