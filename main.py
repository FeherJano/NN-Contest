import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from torchvision import transforms
import os
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
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error

# Eszköz beállítása
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Teszt adatok és konfiguráció
test_data_path = "test_data"
batch_size = 16  # Ezt a konfigurációt testreszabhatja
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Az edzés alatt használt statisztikákat használja
])


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

# Teszt DataLoader létrehozása
all_files = os.listdir(test_data_path)
filename_ids = sorted(set(f.split('_amp')[0].split('_phase')[0].split('_mask')[0] for f in all_files))
test_labels_df = pd.DataFrame({'filename_id': filename_ids})
test_dataset = HoloMicroscopeDataset(test_labels_df, root_dir=test_data_path, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Modell inicializálása és betöltése
model = MultiTaskRegressor().to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

# Prediction
def predict_test_data(model, test_loader, output_file="predictions.csv"):
    model.eval()
    results = []
    with torch.no_grad():
        for images in test_loader:
            images = images.to(device)
            focus_output, _ = model(images)
            predictions = focus_output.view(-1).cpu().round().abs().int().tolist()
            results.extend(predictions)

    # Mentés CSV fájlba
    test_df = pd.DataFrame({'Id': test_loader.dataset.labels_df['filename_id'], 'Expected': results})
    test_df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

# Prediction futtatása
predict_test_data(model, test_loader)
