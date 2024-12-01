import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torchvision import models
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import mean_squared_error

# test_run beállítása
test_run = True

# Adatok és útvonalak beállítása a test_run alapján
if test_run:
    labels_df = pd.read_csv('temp_data_labels.csv')
    train_data_path = 'temp_train_data'
    test_data_path = 'temp_test_data'
    test_solution_path = 'temp_solution.csv'
else:
    labels_df = pd.read_csv('data_labels_train.csv')
    train_data_path = 'train_data'
    test_data_path = 'test_data'
    test_solution_path = None

class HoloMicroscopeDataset(Dataset):
    def __init__(self, labels_df, root_dir, transform=None):
        self.labels_df = labels_df
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        filename_id = row['filename_id']
        defocus_label = row['defocus_label'] if 'defocus_label' in row else None
        class_label = row['class_label'] if 'class_label' in row else None

        # Képek beolvasása
        amp_img = Image.open(os.path.join(self.root_dir, f"{filename_id}_amp.png")).convert("L")
        phase_img = Image.open(os.path.join(self.root_dir, f"{filename_id}_phase.png")).convert("L")
        mask_img = Image.open(os.path.join(self.root_dir, f"{filename_id}_mask.png")).convert("L")

        # Képek összevonása csatornák szerint (amplitúdó, fázis, maszk)
        image = np.stack([np.array(amp_img), np.array(phase_img), np.array(mask_img)], axis=0)

        # Átalakítás PIL formátumra
        image = Image.fromarray(image.transpose(1, 2, 0).astype(np.uint8))

        # Ha van transzformáció, akkor alkalmazzuk
        if self.transform:
            image = self.transform(image)

        if defocus_label is not None and class_label is not None:
            return image, torch.tensor(defocus_label, dtype=torch.float32), torch.tensor(class_label, dtype=torch.long)
        elif defocus_label is not None:
            return image, torch.tensor(defocus_label, dtype=torch.float32)
        else:
            return image


# Transzformációk és augmentációk beállítása
transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Dataset és DataLoader beállítása
train_df, val_df = train_test_split(labels_df, test_size=0.2, random_state=42)
train_dataset = HoloMicroscopeDataset(train_df, root_dir=train_data_path, transform=transform)
val_dataset = HoloMicroscopeDataset(val_df, root_dir=train_data_path, transform=transform)

# Beállítások a megadott batch mérettel és tanulási rátával
batch_size = 8
learning_rate = 0.0001

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# UNet Model for Regression
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        self.enc1 = self.contracting_block(in_channels, 64)
        self.enc2 = self.contracting_block(64, 128)
        self.enc3 = self.contracting_block(128, 256)
        self.enc4 = self.contracting_block(256, 512)
        self.bottleneck = self.contracting_block(512, 1024)
        self.dec4 = self.expanding_block(1024 + 512, 512)
        self.dec3 = self.expanding_block(512 + 256, 256)
        self.dec2 = self.expanding_block(256 + 128, 128)
        self.dec1 = self.expanding_block(128 + 64, 64)
        self.final_layer = nn.Conv2d(64, out_channels, kernel_size=1)

    def contracting_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def expanding_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(nn.functional.max_pool2d(enc1, kernel_size=2))
        enc3 = self.enc3(nn.functional.max_pool2d(enc2, kernel_size=2))
        enc4 = self.enc4(nn.functional.max_pool2d(enc3, kernel_size=2))

        # Bottleneck
        bottleneck = self.bottleneck(nn.functional.max_pool2d(enc4, kernel_size=2))

        # Decoder with concatenation
        dec4 = self.dec4(torch.cat(
            (enc4, nn.functional.interpolate(bottleneck, size=enc4.shape[2:], mode='bilinear', align_corners=False)),
            dim=1))
        dec3 = self.dec3(torch.cat(
            (enc3, nn.functional.interpolate(dec4, size=enc3.shape[2:], mode='bilinear', align_corners=False)), dim=1))
        dec2 = self.dec2(torch.cat(
            (enc2, nn.functional.interpolate(dec3, size=enc2.shape[2:], mode='bilinear', align_corners=False)), dim=1))
        dec1 = self.dec1(torch.cat(
            (enc1, nn.functional.interpolate(dec2, size=enc1.shape[2:], mode='bilinear', align_corners=False)), dim=1))

        # Final layer
        output = self.final_layer(dec1)

        # Global Average Pooling (GAP) to reduce spatial dimensions
        output = output.mean(dim=[2, 3])  # Average over height and width
        return output


# Replace RegressionModel with UNet
regression_model = UNet(in_channels=3, out_channels=1)

# Loss function and optimizer remain unchanged
regression_criterion = nn.MSELoss()
optimizer = torch.optim.Adam(regression_model.parameters(), lr=learning_rate)

num_epochs = 10

# Tanítási ciklus a regressziós modellre
for epoch in range(num_epochs):
    regression_model.train()
    train_focus_loss = 0
    for images, focus_targets, _ in train_loader:  # Csak regressziós címkék kellenek
        optimizer.zero_grad()
        focus_output = regression_model(images)
        focus_loss = regression_criterion(focus_output.view(-1), focus_targets)
        focus_loss.backward()
        optimizer.step()
        train_focus_loss += focus_loss.item()

    # Validációs fázis
    regression_model.eval()
    val_focus_loss = 0
    with torch.no_grad():
        for images, focus_targets, _ in val_loader:
            focus_output = regression_model(images)
            focus_loss = regression_criterion(focus_output.view(-1), focus_targets)
            val_focus_loss += focus_loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Train Focus Loss: {train_focus_loss / len(train_loader)}, "
          f"Val Focus Loss: {val_focus_loss / len(val_loader)}")



# Teszt DataLoader létrehozása
all_files = os.listdir(test_data_path)
filename_ids = sorted(set(f.split('_amp')[0].split('_phase')[0].split('_mask')[0] for f in all_files))
test_labels_df = pd.DataFrame({'filename_id': filename_ids})
test_dataset = HoloMicroscopeDataset(test_labels_df, root_dir=test_data_path, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

def predict_and_save_results(model, test_loader, output_file='predictions.csv'):
    regression_model.eval()
    results = []
    with torch.no_grad():
        for images in test_loader:
            focus_output, _ = model(images)
            predictions = focus_output.view(-1).round().abs().int()
            results.extend(predictions.tolist())

    # Mentés CSV fájlba
    test_df = pd.DataFrame({'Id': test_loader.dataset.labels_df['filename_id'], 'Expected': results})
    test_df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

# Prediction és összehasonlítás a temp_solution.csv-vel, ha test_run igaz
if test_run:
    regression_model.eval()
    predictions = []
    ground_truth = pd.read_csv(test_solution_path)

    with torch.no_grad():
        for images in test_loader:
            focus_output = regression_model(images)
            predictions.extend(focus_output.view(-1).round().abs().int().tolist())

    # Rendezés az `Id` oszlop alapján, hogy megegyezzen a `ground_truth` sorrendjével
    predictions_df = pd.DataFrame({'Id': test_labels_df['filename_id'], 'Prediction': predictions})
    predictions_df = predictions_df.set_index('Id').reindex(ground_truth['Id']).reset_index()

    # Visszaállítjuk a rendezett predikciókat listává
    ordered_predictions = predictions_df['Prediction'].tolist()

    # RMSE kiszámítása
    rmse = mean_squared_error(ground_truth['Expected'], ordered_predictions, squared=False)
    print(f"Root Mean Squared Error (RMSE): {rmse}")

else:
    # Predictions.csv mentése, ha nem teszt módban fut
    predict_and_save_results(regression_model, test_loader)

