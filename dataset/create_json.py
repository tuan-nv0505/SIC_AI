import pandas as pd
import json
from sklearn.model_selection import train_test_split
import os


df = pd.read_csv("dataset/fashion_small/styles.csv", on_bad_lines="skip", nrows=100) #nrows = 100 (100 anh dau)
df = df.dropna(subset=['id', 'gender', 'masterCategory', 'subCategory', 'articleType', 'baseColour'])

train_val, test = train_test_split(df, test_size=0.1, random_state=42)
train, val = train_test_split(train_val, test_size=0.1111, random_state=42)

train['split'] = 'train'
val['split'] = 'val'
test['split'] = 'test'

folder_path = "dataset/fashion_small/resized_images"
with open("dataset/img_test.txt", "w") as file:
    for _, row in test.iterrows():
        img_name = str(row["id"]) + ".jpg"
        if os.path.exists(os.path.join(folder_path, img_name)):
            file.write("{}\n".format(img_name))


#
# df_all = pd.concat([train, val, test], ignore_index=True)
# image_folder = 'dataset/fashion_small/resized_images'
#
# images_list = []
# anh_tao_duoc = 0
# anh_loi = 0
# du_lieu_train, du_lieu_val, du_lieu_test = 0, 0, 0
# for _, row in df_all.iterrows():
#     anh_tao_duoc += 1
#     image_id = str(row['id']) + '.jpg'
#     image_path = os.path.join(image_folder, image_id)
#     if not os.path.exists(image_path):
#         anh_loi += 1
#         print("anh loi: {}, khong the tai anh nay!".format(image_path))
#         continue
#
#     caption_tokens = [str(row[col]).lower() for col in ['gender', 'masterCategory', 'subCategory', 'articleType', 'baseColour']]
#
#     image_dict = {
#         'filename': image_id,
#         'filepath': image_folder,
#         'title': row['productDisplayName'],
#         'split': row['split'],
#         'sentences': [{'tokens': caption_tokens}]
#     }
#     images_list.append(image_dict)
#     if row['split'] == "train": du_lieu_train += 1
#     if row['split'] == "val": du_lieu_val += 1
#     if row['split'] == "test": du_lieu_test += 1
#
# output_json = {"images": images_list}
#
# path_json = "dataset/fashion_small_dataset.json"
# if os.path.exists(path_json):
#   os.remove(path_json)
#
# # luu file json mo ta dataset fashion_small
# with open(path_json, "w") as f:
#     json.dump(output_json, f, indent=4)
#
# print("tao file json thanh cong voi: {} anh!".format(len(images_list)))
# print("anh tao thanh cong: {}".format(anh_tao_duoc))
# print("anh loi: {}".format(anh_loi))
# print("train: {} | val: {} | test: {}".format(du_lieu_train, du_lieu_val, du_lieu_test))
