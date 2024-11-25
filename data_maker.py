import os
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil

# Eredeti adatok beolvasása
labels_df = pd.read_csv('data_labels_train.csv')

# Train és teszt részre bontás
train_df, test_df = train_test_split(labels_df, test_size=0.1, random_state=42)

# Létrehozzuk az új mappákat, ha még nem léteznek
os.makedirs('temp_train_data', exist_ok=True)
os.makedirs('temp_test_data', exist_ok=True)

# Eredeti train_data mappa útvonala
train_data_path = 'train_data'

# Fájlok másolása a temp_train_data mappába
for _, row in train_df.iterrows():
    filename_id = row['filename_id']
    for suffix in ['_amp.png', '_phase.png', '_mask.png']:
        src_file = os.path.join(train_data_path, f"{filename_id}{suffix}")
        dest_file = os.path.join('temp_train_data', f"{filename_id}{suffix}")
        shutil.copy(src_file, dest_file)

# Fájlok másolása a temp_test_data mappába
for _, row in test_df.iterrows():
    filename_id = row['filename_id']
    for suffix in ['_amp.png', '_phase.png', '_mask.png']:
        src_file = os.path.join(train_data_path, f"{filename_id}{suffix}")
        dest_file = os.path.join('temp_test_data', f"{filename_id}{suffix}")
        shutil.copy(src_file, dest_file)

# Létrehozzuk a temp_data_labels.csv fájlt, amely a temp_train_data-hoz tartozik
train_df.to_csv('temp_data_labels.csv', index=False)

# Létrehozzuk a temp_solution.csv fájlt a temp_test_data mappához, kerekítjük és abszolút értéket veszünk
temp_solution_df = test_df[['filename_id', 'defocus_label']].copy()
temp_solution_df['defocus_label'] = temp_solution_df['defocus_label'].round().abs().astype(int)  # Konverzió integer típusra
temp_solution_df.columns = ['Id', 'Expected']  # Az oszlopok elnevezése az elvárt formátum szerint
temp_solution_df.to_csv('temp_solution.csv', index=False)

print("Az adatok sikeresen szétválasztásra kerültek:")
print("- temp_train_data: az eredeti adatok 90%-a")
print("- temp_test_data: az eredeti adatok 10%-a")
print("- temp_data_labels.csv: a temp_train_data megoldásai")
print("- temp_solution.csv: a temp_test_data megoldásai (kerekítve és abszolút egész számként)")
