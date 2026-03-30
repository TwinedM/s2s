import pandas as pd
import os

BASE_DIR = r"C:\Users\pieush\Desktop\Root\Sign-to-Speech\dataset"
OUTPUT_FILE = r"C:\Users\pieush\Desktop\Root\Sign-to-Speech\dataset\master_alphabets_train_2.csv"

all_data = []

# Loop only through folders 001 to 020
for i in range(1, 26):
    folder_name = f"{i:03d}"  # formats as 001, 002, ...
    folder_path = os.path.join(BASE_DIR, folder_name)
    
    alphabet_path = os.path.join(folder_path, "alphabet")
    
    if not os.path.exists(alphabet_path):
        print(f"Skipping {folder_name} (no alphabets folder)")
        continue

    for file in os.listdir(alphabet_path):
        if file.lower().endswith(".csv"):
            
            label = os.path.splitext(file)[0].upper()
            file_path = os.path.join(alphabet_path, file)

            df = pd.read_csv(file_path)
            df["label"] = label

            all_data.append(df)

# Combine everything
master_df = pd.concat(all_data, ignore_index=True)

# Save
master_df.to_csv(OUTPUT_FILE, index=False)

print("✅ Training dataset created")
print("Total rows:", master_df.shape[0])
print("Saved at:", OUTPUT_FILE)