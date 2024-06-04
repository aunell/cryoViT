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
def get_ignore_mask(directory, ignore_paths, height=3010, width=2464):
    masks=[]
    for ignore_path_indexed in ignore_paths:
        print('working')
        ignore_path = os.path.join(directory, ignore_path_indexed)
        print('ignore path:', ignore_path)
        mask_path = ignore_path.split('.')[0] + '.pt'  # Path where the mask will be saved
        if os.path.exists(mask_path):
            # If the mask file exists, load it
            ignore_mask = torch.load(mask_path).cuda()
        else:
            # If the mask file doesn't exist, create the mask
            with open(ignore_path, 'rb') as f:
                ignore_coords = list(set(pickle.load(f)))
            ignore_mask = torch.zeros((height,width)).cuda()
            for y, x in tqdm(ignore_coords):
                try:
                    ignore_mask[x, y] = 1
                except:
                    continue
            del ignore_coords
            # Save the mask for future use
            torch.save(ignore_mask, mask_path)
        masks.append(ignore_mask)
    return masks
class SegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, ignore_dir, feature_dir, device, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.ignore_dir = ignore_dir
        self.feature_dir = feature_dir
        self.transform = transform
        self.device = device

        self.img_names = sorted(os.listdir(img_dir))
        self.mask_names = sorted(os.listdir(mask_dir))
        self.ignore_names = sorted(os.listdir(ignore_dir))
        self.feature_names = sorted(os.listdir(feature_dir))

        self.ignore_masks=get_ignore_mask('/pasteur/u/aunell/cryoViT/data/training/ignore_pt', self.ignore_names)

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

        return {'image': img, 'features': features, 'mask': mask, 'ignore_mask': self.ignore_masks[idx]}
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
            # ignore_coords = batch['ignore_coords']
            # ignore_mask = create_ignore_mask(ignore_coords, masks_raw)
            ignore_mask = batch['ignore_mask']
            image_predicted = torch.zeros_like(masks_raw)
            for i in range(features_raw.shape[2]):
                for j in range(features_raw.shape[3]):
                    features = features_raw[:, :, i, j].squeeze()
                    # features = features.view(features.size(0), -1).type(torch.float32).transpose(0, 1)
                    preds = model(features.float()).unsqueeze(0).unsqueeze(0)
                    image_predicted[0,0,i*14:(i+1)*14, j*14:(j+1)*14] = preds
            loss= calculate_loss(criterion, image_predicted, masks_raw, ignore_mask)
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        epoch_loss = running_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
    print("Training complete.")

# def calculate_loss(criterion, outputs, mask, ignore_mask):
#     logits_mask = outputs

#     logits_mask = logits_mask * (1 - ignore_mask)
#     mask = mask * (1 - ignore_mask)

#     # Calculate the loss    
#     print(logits_mask.shape, mask.shape)
#     loss= criterion(logits_mask.cuda(), mask.cuda())  
#     tp, fp, fn, tn = smp.metrics.get_stats(logits_mask.long(), mask.long(), mode="binary")
#     print('tp:', tp, 'fp:', fp, 'fn:', fn, 'tn:', tn)
#     return loss
def calculate_loss(criterion, outputs, mask, ignore_mask):
    import torch.nn.functional as F
    from torchvision.utils import save_image
    mask = mask.squeeze().squeeze()
    logits_mask = outputs
    print('logits_mask shape:', logits_mask.shape)
    # logits_mask = torch.sigmoid(logits_mask).unsqueeze(0).unsqueeze(0)
    # logits_mask = (logits_mask >= 0.5).float()
    logits_mask = F.logsigmoid(logits_mask).exp()
    # logits_mask = (logits_mask >= 0.5).float().requires_grad_()
    logits_mask = logits_mask*(1 - ignore_mask)
    mask = mask * (1 - ignore_mask)
    print('mask shape', mask.shape)
    save_image(logits_mask.squeeze().cpu(), 'output.png')
    save_image(mask.unsqueeze(0).cpu(), 'output_mask.png')
    save_image(ignore_mask.unsqueeze(0).cpu(), 'output_ignore.png')

    # Calculate the loss
    print('mask shape', mask.shape)
    mask = mask.unsqueeze(0).unsqueeze(0)
    criterion = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=False)  # dice_loss
    # criterion= dice_loss
    loss = criterion(logits_mask.cuda(), mask.cuda())
    print('loss:', loss.item())
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
            preds = np.where(preds > 0.65, 1, 0)
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
            concatenated.save(f'/pasteur/u/aunell/cryoViT/segmentation/results0531/linear_{b}{i}.png')

# Define the neural network
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(1536, 196)  # Fully connected layer to map input to 196 units
        # self.relu = nn.ReLU()  
        # self.fc1 = nn.Conv2d(1536, 1, 1)
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
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=7)
    print('dataloader loaded')
    model = MyModel().to(device)
    criterion = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    # model.load_state_dict(torch.load('/pasteur/u/aunell/cryoViT/segmentation/linear_model.pth'))
    train_model(model, dataloader, criterion, optimizer, num_epochs=2)
    # torch.save(model.state_dict(), '/pasteur/u/aunell/cryoViT/segmentation/linear_model.pth')
    test_dataset = SegmentationDataset(img_dir, mask_dir, ignore_dir, feature_dir, device)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=7)
    visualize_predictions(model, test_dataloader)