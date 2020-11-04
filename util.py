# This file contains helper methods for the demo notebook
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import cv2

BDD_CLASSES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
    'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
    'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'void'
]
BDD_IDS = list(range(len(BDD_CLASSES) - 1)) + [255]
BDD_ID_MAP = {
    id:ndx
    for ndx, id in enumerate(BDD_IDS)
}

n_classes = len(BDD_CLASSES)
bdd_dir = os.path.join('.', 'bdd100k', 'seg')
train_dir = os.path.join(bdd_dir, 'images', 'train')
color_labels_dir = os.path.join(bdd_dir, 'color_labels', 'train')
labels_dir = os.path.join(bdd_dir, 'labels', 'train')

train_ids = [name.split(".")[0] for name in os.listdir(train_dir)]

def download_data():
    if not os.path.exists("bdd100k.tgz"):
        print("Downloading data from https://storage.googleapis.com/l2kzone/bdd100k.tgz...")
        os.system('curl https://storage.googleapis.com/l2kzone/bdd100k.tgz --output bdd100k.tgz')
    
    if not os.path.exists("bdd100k"):
        print("Extracting data...")
        os.system('tar xzf bdd100k.tgz')
    
    print("Raw data downlaoded to ./bdd100k.")

def show_image(path):
    plt.imshow(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB))
    plt.show()
    
def get_train_image_path(ndx):
    return os.path.join(train_dir, train_ids[ndx] + ".jpg")

def get_color_label_image_path(ndx):
    return os.path.join(color_labels_dir, train_ids[ndx] + "_train_color.png")

def get_label_image_path(ndx):
    return os.path.join(labels_dir, train_ids[ndx] + "_train_id.png")

def get_dominant_id_ndx(np_image):
    if isinstance(np_image, wandb.Image):
        np_image = np.array(np_image._image)
    return BDD_ID_MAP[np.argmax(np.bincount(np_image.astype(int).flatten()))]

def downsample_image(image, factor=2):
    return image[::factor, ::factor]

def downscale_image(image, factor=2):
    return cv2.resize(image, downsample_image(image, factor).shape[:2][::-1])

def clean_artifacts_dir():
    if os.path.isdir("artifacts"):
        shutil.rmtree("artifacts")
        
def mask_to_bounding(np_image):
    if isinstance(np_image, wandb.Image):
        np_image = np.array(np_image._image)
    
    data = []
    for id_num in BDD_IDS:
        matches = np_image == id_num
        col_count = np.where(matches.sum(axis=0))[0]
        row_count = np.where(matches.sum(axis=1))[0]
        
        if len(col_count) > 1 and len(row_count) > 1:
            minX = col_count[0] / np_image.shape[1]
            maxX = col_count[-1] / np_image.shape[1]
            minY = row_count[0] / np_image.shape[0]
            maxY = row_count[-1] / np_image.shape[0]
        
            data.append({
                  "position": {
                    "minX": minX,
                    "maxX": maxX,
                    "minY": minY,
                    "maxY": maxY,
                },
                "class_id" : id_num,          
            })
    return data