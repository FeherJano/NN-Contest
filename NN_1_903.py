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

#------Config------
batch_size = 8
learning_rate = 0.0001
num_epochs = 300  # Epoch száma


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

# Calculate Dataset Statistics (Mean and Std)
def calculate_mean_std(root_dir, labels_df):
    """Compute the mean and standard deviation of the dataset."""
    sample_images = []
    for idx, row in labels_df.iterrows():
        filename_id = row['filename_id']
        amp_img = np.array(Image.open(os.path.join(root_dir, f"{filename_id}_amp.png")).convert("L"))
        phase_img = np.array(Image.open(os.path.join(root_dir, f"{filename_id}_phase.png")).convert("L"))
        mask_img = np.array(Image.open(os.path.join(root_dir, f"{filename_id}_mask.png")).convert("L"))
        stacked_image = np.stack([amp_img, phase_img, mask_img], axis=2)
        sample_images.append(stacked_image)
    sample_images = np.array(sample_images) / 255.0  # Normalize to [0, 1] before computing stats
    mean = sample_images.mean(axis=(0, 1, 2))
    std = sample_images.std(axis=(0, 1, 2))
    return mean, std

train_mean, train_std = calculate_mean_std(train_data_path, labels_df)

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


# Transformations
transform_train = transforms.Compose([
    transforms.RandomRotation(degrees=10),
    transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=train_mean.tolist(), std=train_std.tolist())
])

transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=train_mean.tolist(), std=train_std.tolist())
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=train_mean.tolist(), std=train_std.tolist())
])


# Split data and create datasets
train_df, val_df = train_test_split(labels_df, test_size=0.2, random_state=42)
train_dataset = HoloMicroscopeDataset(train_df, root_dir=train_data_path, transform=transform_train)
val_dataset = HoloMicroscopeDataset(val_df, root_dir=train_data_path, transform=transform_val)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# Multi-task modell: ResNet-18 alapú háló
class MultiTaskRegressor(nn.Module):
    def __init__(self):
        super(MultiTaskRegressor, self).__init__()
        self.backbone = models.resnet18(weights="ResNet18_Weights.IMAGENET1K_V1")
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 128)  # ResNet-18 backbone outputs features

        # Intermediate task-specific layers
        self.shared = nn.Linear(128, 64)
        self.fc_focus = nn.Sequential(
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Focus distance regression
        )
        self.fc_class = nn.Sequential(
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4)  # Class classification
        )

    def forward(self, x):
        features = torch.relu(self.backbone(x))  # Extracted features
        shared_out = self.shared(features)  # Shared layer
        focus_output = self.fc_focus(shared_out)
        class_output = self.fc_class(shared_out)
        return focus_output, class_output




model = MultiTaskRegressor()
focus_criterion = nn.SmoothL1Loss()
class_criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = CosineAnnealingLR(optimizer, T_max=10)


# Tanítási ciklus az új hyperparaméterekkel
for epoch in range(num_epochs):
    model.train()
    train_focus_loss = 0
    train_class_loss = 0
    for images, focus_targets, class_targets in train_loader:
        optimizer.zero_grad()
        focus_output, class_output = model(images)
        focus_loss = focus_criterion(focus_output.view(-1), focus_targets)
        class_loss = class_criterion(class_output, class_targets)
        loss = focus_loss + class_loss
        loss.backward()
        optimizer.step()
        train_focus_loss += focus_loss.item()
        train_class_loss += class_loss.item()

    scheduler.step()

    # Validációs fázis
    model.eval()
    val_focus_loss = 0
    val_class_loss = 0
    with torch.no_grad():
        for images, focus_targets, class_targets in val_loader:
            focus_output, class_output = model(images)
            focus_loss = focus_criterion(focus_output.view(-1), focus_targets)
            class_loss = class_criterion(class_output, class_targets)
            val_focus_loss += focus_loss.item()
            val_class_loss += class_loss.item()

    print(
        f"Epoch {epoch + 1}/{num_epochs}, Train Focus Loss: {train_focus_loss / len(train_loader)}, "
        f"Train Class Loss: {train_class_loss / len(train_loader)}, "
        f"Val Focus Loss: {val_focus_loss / len(val_loader)}, Val Class Loss: {val_class_loss / len(val_loader)}")

# Teszt DataLoader létrehozása
all_files = os.listdir(test_data_path)
filename_ids = sorted(set(f.split('_amp')[0].split('_phase')[0].split('_mask')[0] for f in all_files))
test_labels_df = pd.DataFrame({'filename_id': filename_ids})
test_dataset = HoloMicroscopeDataset(test_labels_df, root_dir=test_data_path, transform=transform_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

def predict_and_save_results(model, test_loader, output_file='predictions.csv'):
    model.eval()
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
    model.eval()
    predictions = []
    ground_truth = pd.read_csv(test_solution_path)

    with torch.no_grad():
        for images in test_loader:
            focus_output, _ = model(images)
            predictions.extend(focus_output.view(-1).round().abs().int().tolist())

    # Rendezés az `Id` oszlop alapján, hogy megegyezzen a `ground_truth` sorrendjével
    predictions_df = pd.DataFrame({'Id': test_labels_df['filename_id'], 'Prediction': predictions})
    predictions_df = predictions_df.set_index('Id').reindex(ground_truth['Id']).reset_index()

    # Visszaállítjuk a rendezett predikciókat listává
    ordered_predictions = predictions_df['Prediction'].tolist()

    # RMSE kiszámítása
    rmse = mean_squared_error(ground_truth['Expected'], ordered_predictions, squared=False)
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    predict_and_save_results(model, test_loader)

else:
    # Predictions.csv mentése, ha nem teszt módban fut
    predict_and_save_results(model, test_loader)

