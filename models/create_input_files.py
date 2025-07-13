from utils import create_input_files
import os
import shutil

if __name__ == '__main__':
    if os.path.exists("output"):
        shutil.rmtree("output")

    os.makedirs("output")

    create_input_files(dataset='atlas',
                       karpathy_json_path = 'dataset/fashion_small_dataset.json',
                       image_folder='dataset/fashion_small/resized_images',
                       captions_per_image=1,
                       min_word_freq=1,
                       output_folder='output',
                       max_len=50)
