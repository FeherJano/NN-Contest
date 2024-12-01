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


# Regresszióért felelős modell
class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.backbone = models.resnet18(weights="ResNet18_Weights.IMAGENET1K_V1")
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone.fc = nn.Identity()  # ResNet kimenetét feldolgozzuk különálló rétegekkel

        # Regressziós rétegek
        self.fc1 = nn.Linear(512, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc_output = nn.Linear(128, 1)

    def forward(self, x):
        x = self.backbone(x)  # ResNet-18 jellemzők
        x = torch.relu(self.bn1(self.fc1(x)))  # Első teljes összekötött réteg
        x = self.dropout1(x)
        x = torch.relu(self.bn2(self.fc2(x)))  # Második teljes összekötött réteg
        x = self.dropout2(x)
        x = self.fc_output(x)  # Regressziós kimenet
        return x


# Osztályozásért felelős modell (opcionális, külön tanítható, ha szükséges)
class ClassificationModel(nn.Module):
    def __init__(self):
        super(ClassificationModel, self).__init__()
        self.backbone = models.resnet18(weights="ResNet18_Weights.IMAGENET1K_V1")
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone.fc = nn.Identity()  # ResNet kimenetét feldolgozzuk különálló rétegekkel

        # Osztályozási rétegek
        self.fc1 = nn.Linear(512, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc_output = nn.Linear(128, 4)  # Négy osztályra osztályoz

    def forward(self, x):
        x = self.backbone(x)  # ResNet-18 jellemzők
        x = torch.relu(self.bn1(self.fc1(x)))  # Első teljes összekötött réteg
        x = self.dropout1(x)
        x = torch.relu(self.bn2(self.fc2(x)))  # Második teljes összekötött réteg
        x = self.dropout2(x)
        x = self.fc_output(x)  # Osztályozási kimenet
        return x


# Loss funkciók és tanítási folyamat
regression_model = RegressionModel()
regression_criterion = nn.SmoothL1Loss()  # Regressziós veszteség
optimizer = torch.optim.Adam(regression_model.parameters(), lr=0.0001)

classification_model = ClassificationModel()  # Ha használni akarod
classification_criterion = nn.CrossEntropyLoss()
classification_optimizer = torch.optim.Adam(classification_model.parameters(), lr=0.0001)

num_epochs = 2

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

