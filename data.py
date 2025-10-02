# import pandas as pd


# data = pd.read_csv("/Users/personal/Desktop/butterfly/test")
# print(data.head(5))
# print(data.__len__())
# # data = pd.read_csv("/Users/personal/Desktop/butterfly/labels.csv")
import pandas as pd

print(pd.read_csv("/Users/personal/Desktop/butterfly/Testing_set.csv").head())
print(pd.read_csv("/Users/personal/Desktop/butterfly/Validation_set.csv").head())


# import pandas as pd
# import os
# import shutil
# from sklearn.model_selection import train_test_split

# # paths
# base_dir = "/Users/personal/Desktop/butterfly"
# train_csv = os.path.join(base_dir, "Training_set.csv")
# train_folder = os.path.join(base_dir, "train")
# val_folder = os.path.join(base_dir, "validation")

# # make validation folder if not exists
# os.makedirs(val_folder, exist_ok=True)

# # read CSV
# df = pd.read_csv(train_csv)

# # split into train (90%) and val (10%) with stratified sampling
# train_df, val_df = train_test_split(
#     df, test_size=0.1, stratify=df["label"], random_state=42
# )

# # --- Save new CSVs ---
# train_df.to_csv(os.path.join(base_dir, "Train_split.csv"), index=False)
# val_df.to_csv(os.path.join(base_dir, "Validation_set.csv"), index=False)

# # --- Copy validation images ---
# for fname in val_df["filename"]:
#     src = os.path.join(train_folder, fname)
#     dst = os.path.join(val_folder, fname)
#     shutil.copy(src, dst)

# print(f"✅ Train size: {len(train_df)}, Validation size: {len(val_df)}")
# print(f"Validation images saved in: {val_folder}")
