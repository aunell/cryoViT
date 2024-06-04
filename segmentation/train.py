import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
from conv_0530 import fix_mask_dimensions, calculate_loss, get_ignore_mask
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import numpy as np
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
import torch.multiprocessing as mp
import wandb
import argparse


# define the LightningModule
class SegmentationModel(L.LightningModule):
    def __init__(self, log_status):
        super().__init__()
        self.log_status=log_status
        self.model = nn.ConvTranspose2d(
                    in_channels=1536, 
                    out_channels=1, 
                    kernel_size=14, 
                    stride=14
                )


    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        features= batch['features'] #.squeeze()
        masks_raw = batch['mask'] #.squeeze().squeeze()
        masks = fix_mask_dimensions(masks_raw, features)
        features=features #.unsqueeze(0)
         # Forward pass
        # print('features shape', features.shape)
        outputs = self.model(features.float())
        # print('outputs shape', outputs.shape)
        ignore_mask = batch['ignore_mask']
        loss= calculate_loss(criterion, outputs, masks, ignore_mask)
        print(f"Epoch {self.current_epoch}, Batch {batch_idx}, Loss: {loss}")
        # Logging to TensorBoard (if installed) by default
        if self.log_status=="True":
            wandb.log({"train_loss": loss})
        if self.current_epoch % 10 == 0:  # Change this condition as needed
                visualize_predictions(self.model, batch, self.log_status)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
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

        return {'image': img, 'features': features, 'mask': mask, 'ignore_mask': self.ignore_masks[idx]}
                # 'ignore_coords': ignore_path}


def visualize_predictions(model, batch, log_status):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Do not calculate gradients
            features_raw = batch['features']
            masks_raw = batch['mask']
            masks_raw = fix_mask_dimensions(masks_raw, features_raw)
            image_predicted=model(features_raw.float())
            image_predicted=torch.sigmoid(image_predicted)
            # Move images, labels and preds to cpu for visualization
            for idx in range(batch['features'].shape[0]):
                images = batch['image'][idx].cpu().numpy()
                labels = masks_raw[idx].cpu().numpy()
                preds = image_predicted[idx].cpu().numpy()

                padding = np.abs(images.shape[0] - preds.shape[1])
                image_transformed = np.vstack((images, np.zeros((padding, images.shape[1]))))

                image_pil=Image.fromarray((image_transformed* 255).astype(np.uint8))
                labels_np = (labels * 255).astype(np.uint8)
                preds_np = (preds * 255).astype(np.uint8).reshape(3010, 2464)

                labels_pil = Image.fromarray(labels_np.squeeze())
                preds_pil = Image.fromarray(preds_np)
                

                # Concatenate the images
                concatenated = np.concatenate((image_pil, labels_pil, preds_pil), axis=1)
                concatenated = Image.fromarray(concatenated)
                if log_status=="True":
                    images = wandb.Image(concatenated, caption="")

                    wandb.log({"examples": images})
                else:
                    concatenated.save(f'/pasteur/u/aunell/cryoViT/segmentation/results0604/Conv1layer_{idx}.png')
def parse_arguments():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('--log_status', type=str, default="False", help='The output directory')
    args = parser.parse_args()
    return args
# init the autoencoder
if __name__ == "__main__":
    args=parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mp.set_start_method('spawn')
    img_dir = '/pasteur/u/aunell/cryoViT/data/sample_data/original'
    mask_dir = '/pasteur/u/aunell/cryoViT/data/training/mask'
    ignore_dir = '/pasteur/u/aunell/cryoViT/data/training/ignore'
    feature_dir = '/pasteur/u/aunell/cryoViT/data/training/features'
    dataset = SegmentationDataset(img_dir, mask_dir, ignore_dir, feature_dir, device)

    criterion = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=False)
    segmentation_model = SegmentationModel(args.log_status)
    checkpoint_callback = ModelCheckpoint(monitor='val_loss',dirpath='/pasteur/u/aunell/cryoViT/segmentation/',filename='convLightning-{epoch:02d}-{val_loss:.2f}', every_n_epochs=500)
    if args.log_status == "True":
        wandb_logger = WandbLogger(log_model="all", project="cryoViT")
        trainer = L.Trainer(logger=wandb_logger, max_epochs=1500) #, callbacks=[checkpoint_callback], evaluation_strategy="steps")
    else:
        trainer = L.Trainer(max_epochs=1500, callbacks=[checkpoint_callback])
    train_loader = DataLoader(dataset, batch_size=7, shuffle=True, num_workers=7)
    trainer.fit(model=segmentation_model, train_dataloaders=train_loader)