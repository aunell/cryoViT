from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from threading import Lock

lock = Lock()

class SegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, ignore_dir, feature_dir, device, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.ignore_dir = ignore_dir
        self.feature_dir = feature_dir
        self.transform = transform
        self.device = device

        self.img_names = sorted(os.listdir(img_dir))[:1]
        self.mask_names = sorted(os.listdir(mask_dir))[:1]
        self.ignore_names = sorted(os.listdir(ignore_dir))[:1]
        self.feature_names = sorted(os.listdir(feature_dir))[:1]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        feature_path = os.path.join(self.feature_dir, self.feature_names[idx])
        features = torch.from_numpy(torch.load(feature_path).transpose(2,0,1)).to(self.device)
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        img = torch.from_numpy(np.array(Image.open(img_path).convert('L'))/255.0).to(self.device)
        # img = np.array(img.resize((img.width//14, img.height//14)))
        mask_path = os.path.join(self.mask_dir, self.mask_names[idx])
        mask = torch.from_numpy(np.array(Image.open(mask_path).convert("L"))/255.0).to(self.device)
        mask=mask[np.newaxis, :, :]
        mask = mask.to(dtype=torch.float) #torch.tensor(mask, dtype=torch.float)
        ignore_path = os.path.join(self.ignore_dir, self.ignore_names[idx])
        print('got item')

        return {'image': img, 'features': features, 'mask': mask, 'ignore_coords': ignore_path}
def fix_mask_dimensions(mask, features):
    goal_height = features.size(2)*14
    goal_width = features.size(3)*14
    if mask.size(2)!=goal_height:
        add_dimensions = goal_height - mask.size(2)
        zeros = torch.zeros((1, 1, add_dimensions, mask.size(3)), dtype=mask.dtype, device=mask.device)
        mask = torch.cat((mask, zeros), dim=2)
    if mask.size(3)!=goal_width:
        add_dimensions = goal_width - mask.size(3)
        zeros = torch.zeros((1, 1, mask.size(2), add_dimensions), dtype=mask.dtype, device=mask.device)
        mask = torch.cat((mask, zeros), dim=3)
    return mask

# Training loop
def train_model(model, dataloader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        print('epoch:', epoch)
        for batch in dataloader:
            features_raw = batch['features']
            masks_raw = batch['mask']
            masks_raw = fix_mask_dimensions(masks_raw, features_raw)
            ignore_coords = batch['ignore_coords']
            ignore_mask = create_ignore_mask(ignore_coords, masks_raw)
            for i in tqdm(range(features_raw.shape[2])):
                # print('epoch', epoch, 'i', i)
                for j in range(features_raw.shape[3]):
                    features = features_raw[:, :, i, j].squeeze()
                    masks = masks_raw[:, :, i*14:(i+1)*14, j*14:(j+1)*14].squeeze().squeeze()
                    features = features.view(features.size(0), -1).type(torch.float32).transpose(0, 1)

                    # Forward pass
                    outputs = model(features)
                    loss= calculate_loss(i,j,criterion, outputs, masks, ignore_mask)
                    # Backward pass and optimization
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
        
        epoch_loss = running_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
    print("Training complete.")

def calculate_loss(i,j,criterion, outputs, mask, ignore_mask):
    logits_mask = outputs
    ignore_mask = ignore_mask[i*14:(i+1)*14, j*14:(j+1)*14]

    logits_mask = logits_mask * (1 - ignore_mask)
    mask = mask * (1 - ignore_mask)

    # Calculate the loss
    loss= criterion(outputs.cuda(), mask.cuda())  

    return loss


def create_ignore_mask(ignore_path, masks_raw):
    with open(ignore_path[0], 'rb') as f:
        ignore_coords = list(set(pickle.load(f)))
    ignore_mask = torch.zeros_like(masks_raw).squeeze().squeeze()
    for y, x in ignore_coords:
        try:
            ignore_mask[x.item(), y.item()] = 1
        except:
            continue
    del ignore_coords
    return ignore_mask

def visualize_predictions(model, dataloader):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Do not calculate gradients
        for b, batch in enumerate(dataloader):
            features_raw = batch['features']
            masks_raw = batch['mask']
            masks_raw = fix_mask_dimensions(masks_raw, features_raw).squeeze().squeeze()
            image_predicted = torch.zeros_like(masks_raw)
            for i in range(features_raw.shape[2]):
                for j in range(features_raw.shape[3]):
                    features = features_raw[:, :, i, j].squeeze()
                    features = features.view(features.size(0), -1).type(torch.float32).transpose(0, 1)
                    preds = model(features)
                    image_predicted[i*14:(i+1)*14, j*14:(j+1)*14] = preds
            # Move images, labels and preds to cpu for visualization
            images = batch['image'].cpu().numpy()
            labels = masks_raw.cpu().numpy()
            preds = image_predicted.cpu().numpy()
            preds = (preds - preds.min()) / (preds.max() - preds.min())
            # preds = np.where(preds > 0.75, 1, 0)
            padding = np.abs(images[0].shape[0] - preds.shape[0])
            image_tranformed = np.vstack((images[0], np.zeros((padding, images[0].shape[1]))))
            image_pil=Image.fromarray((image_tranformed* 255).astype(np.uint8))
            labels_np = (labels * 255).astype(np.uint8)
            preds_np = (preds * 255).astype(np.uint8)
            labels_pil = Image.fromarray(labels_np)
            preds_pil = Image.fromarray(preds_np)

            # Concatenate the images
            concatenated = np.concatenate((image_pil, labels_pil, preds_pil), axis=1)
            concatenated = Image.fromarray(concatenated)
            concatenated.save(f'/pasteur/u/aunell/cryoViT/segmentation/temp_results/ztest_res_{b}{i}.png')

# Define the neural network
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(1536, 196)  # Fully connected layer to map input to 196 units
        # self.relu = nn.ReLU()  
        self.reshape = lambda x: x.view(14, 14)  # Reshape output to (14, 14)

    def forward(self, x):
        x = self.fc1(x)
        # x = self.relu(x) 
        x = self.reshape(x)
        return x


# Example usage
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mp.set_start_method('spawn')
    img_dir = '/pasteur/u/aunell/cryoViT/data/sample_data/original'
    mask_dir = '/pasteur/u/aunell/cryoViT/data/training/mask'
    ignore_dir = '/pasteur/u/aunell/cryoViT/data/training/ignore'
    feature_dir = '/pasteur/u/aunell/cryoViT/data/training/features'

    dataset = SegmentationDataset(img_dir, mask_dir, ignore_dir, feature_dir, device)
    print('len dataset:', len(dataset))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    print('dataloader loaded')
    model = MyModel().to(device)
    criterion = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.load_state_dict(torch.load('/pasteur/u/aunell/cryoViT/segmentation/linear_model_overfit_all.pth'))
    # train_model(model, dataloader, criterion, optimizer, num_epochs=1)
    # torch.save(model.state_dict(), '/pasteur/u/aunell/cryoViT/segmentation/linear_model_relu.pth')
    test_dataset = SegmentationDataset(img_dir, mask_dir, ignore_dir, feature_dir, device)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=2)
    visualize_predictions(model, test_dataloader)