import pandas as pd
import json
import os
import re
from sklearn.model_selection import train_test_split

df = pd.read_csv("dataset/fashion_small/styles.csv", on_bad_lines="skip")
df = df.dropna(subset=['id', 'gender', 'masterCategory', 'subCategory', 'articleType', 'baseColour', 'productDisplayName'])
print("truoc khi loc:")
print("so anh:", len(df))
print("so articleType:", df['articleType'].nunique())

df = df[df['articleType'].map(df['articleType'].value_counts()) >= 50]
print("sau khi loc")
print("so anh:", len(df))
print("so articleType:", df['articleType'].nunique())

train_val, test = train_test_split(df, test_size=0.1, random_state=42, stratify=df['articleType'])
train, val = train_test_split(train_val, test_size=0.1111, random_state=42, stratify=train_val['articleType'])
print("train")
print(train["articleType"].value_counts())
print("val")
print(val["articleType"].value_counts())
print("test")
print(test["articleType"].value_counts())

train['split'] = 'train'
val['split'] = 'val'
test['split'] = 'test'

folder_path = "dataset/fashion_small/resized_images"
with open("dataset/img_test.txt", "w") as file:
    for _, row in test.iterrows():
        img_name = str(row["id"]) + ".jpg"
        if os.path.exists(os.path.join(folder_path, img_name)):
            file.write("{}\n".format(img_name))

df_all = pd.concat([train, val, test], ignore_index=True)

image_folder = 'dataset/fashion_small/resized_images'
images_list = []

count_total, count_valid, count_error = 0, 0, 0

for _, row in df_all.iterrows():
    count_total += 1
    image_id = str(row['id']) + ".jpg"
    image_path = os.path.join(image_folder, image_id)

    if not os.path.exists(image_path):
        count_error += 1
        continue

    gender = str(row['gender']).lower()
    article_type = str(row['articleType']).lower()
    base_colour = str(row['baseColour']).lower()
    master_category = str(row['masterCategory']).lower()

    caption_tokens = [gender, article_type, base_colour, master_category]
    if len(caption_tokens) == 0:
        continue

    image_dict = {
        'filename': image_id,
        'filepath': image_folder,
        'title': row['productDisplayName'],
        'split': row['split'],
        'sentences': [{'tokens': caption_tokens}]
    }
    images_list.append(image_dict)
    count_valid += 1

output_json = {"images": images_list}
path_json = "dataset/fashion_small_dataset.json"

if os.path.exists(path_json):
    os.remove(path_json)

with open(path_json, "w") as f:
    json.dump(output_json, f, indent=4)

print("tao file json thanh cong!")
print(f"tong anh: {count_total}, thanh cong: {count_valid}, loi: {count_error}")
print("Train/Val/Test:")
print(df_all['split'].value_counts())
