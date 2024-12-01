import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

labels_df = pd.read_csv('data_labels_train.csv')
print(labels_df.head())

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

        amp_img = Image.open(os.path.join(self.root_dir, f"{filename_id}_amp.png")).convert("L")
        phase_img = Image.open(os.path.join(self.root_dir, f"{filename_id}_phase.png")).convert("L")
        mask_img = Image.open(os.path.join(self.root_dir, f"{filename_id}_mask.png")).convert("L")

        image = np.stack([np.array(amp_img), np.array(phase_img), np.array(mask_img)], axis=0)

        image = torch.tensor(image, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        if defocus_label is not None:
            return image, torch.tensor(defocus_label, dtype=torch.float32)
        else:
            return image


transform = transforms.Compose([
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_df, val_df = train_test_split(labels_df, test_size=0.2, random_state=42)
train_dataset = HoloMicroscopeDataset(train_df, root_dir='train_data', transform=transform)
val_dataset = HoloMicroscopeDataset(val_df, root_dir='train_data', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

for images, labels in train_loader:
    print("Batch of images shape:", images.shape)
    print("Batch of labels shape:", labels.shape)
    print("First batch of labels:", labels[:10])
    break


class FocusDistanceRegressor(nn.Module):
    def __init__(self):
        super(FocusDistanceRegressor, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)  # Adjust input size if necessary
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = FocusDistanceRegressor()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 2

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for images, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs.view(-1), targets.float())
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, targets in val_loader:
            outputs = model(images)
            loss = criterion(outputs.view(-1), targets.float())
            val_loss += loss.item()

    print(
        f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss / len(train_loader)}, Val Loss: {val_loss / len(val_loader)}")

test_data_path = 'test_data'
all_files = os.listdir(test_data_path)
filename_ids = sorted(set(f.split('_amp')[0].split('_phase')[0].split('_mask')[0] for f in all_files))
test_labels_df = pd.DataFrame({'filename_id': filename_ids})

test_dataset = HoloMicroscopeDataset(test_labels_df, root_dir='test_data', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


def predict_and_save_results(model, test_loader, output_file='predictions.csv'):
    model.eval()
    results = []
    with torch.no_grad():
        for images in test_loader:
            outputs = model(images)
            print(outputs)
            predictions = outputs.view(-1).round().abs().int()
            results.extend(predictions.tolist())

    test_df = pd.DataFrame({'Id': test_loader.dataset.labels_df['filename_id'], 'Expected': results})
    test_df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

predict_and_save_results(model, test_loader)

