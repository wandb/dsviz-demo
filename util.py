# This file contains helper methods for the demo notebook
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import wandb

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

train_ids = None

def download_data():
    global train_ids
    if not os.path.exists("bdd100k.tgz"):
        print("Downloading data from https://storage.googleapis.com/l2kzone/bdd100k.tgz...")
        os.system('curl https://storage.googleapis.com/l2kzone/bdd100k.tgz --output bdd100k.tgz')
    
    if not os.path.exists("bdd100k"):
        print("Extracting data...")
        os.system('tar xzf bdd100k.tgz')
        
    train_ids = [name.split(".")[0] for name in os.listdir(train_dir)]
    
    print("Raw data downlaoded to ./bdd100k.")

def show_image(path):
    plt.imshow(Image.open(path))
    plt.show()

def _check_train_ids():
    if train_ids is None:
        raise Exception("Please download the data using util.download_data() before attempting to access it.")
    
def get_train_image_path(ndx):
    _check_train_ids()
    return os.path.join(train_dir, train_ids[ndx] + ".jpg")

def get_color_label_image_path(ndx):
    _check_train_ids()
    return os.path.join(color_labels_dir, train_ids[ndx] + "_train_color.png")

def get_label_image_path(ndx):
    _check_train_ids()
    return os.path.join(labels_dir, train_ids[ndx] + "_train_id.png")

def get_dominant_id_ndx(np_image):
    if isinstance(np_image, wandb.Image):
        np_image = np.array(np_image._image)
    return BDD_ID_MAP[np.argmax(np.bincount(np_image.astype(int).flatten()))]

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

def get_scaled_train_image(ndx, factor=2):
    return Image.open(get_train_image_path(ndx)).reduce(factor)

def get_scaled_mask_label(ndx, factor=2):
    return np.array(Image.open(get_label_image_path(ndx)).reduce(factor))

def get_scaled_bounding_boxes(ndx, factor=2):
    return mask_to_bounding(np.array(Image.open(get_label_image_path(ndx)).reduce(factor)))

def get_scaled_color_mask(ndx, factor=2):
    return Image.open(get_color_label_image_path(ndx)).reduce(factor)

def get_dominant_class(label_mask):
    return BDD_CLASSES[get_dominant_id_ndx(label_mask)]

## Model training logic:
import numpy as np
import pickle

class ExampleSegmentationModel:
    def __init__(self, n_classes):
        self.n_classes = n_classes
        
    def train(self, images, masks):
        self.min = images.min()
        self.max = images.max()
        images = (images - self.min) / (self.max - self.min)
        step = 1.0 / n_classes
        self.quantiles = list(np.quantile(images, [i * step for i in range(self.n_classes)]))
        self.quantiles.append(1.0)
        self.outshape = masks.shape
        
    def predict(self, images):
        results = np.zeros((images.shape[0], self.outshape[1], self.outshape[2]))
        images = ((images - self.min) / (self.max - self.min)).mean(axis=3)
        for i in range(self.n_classes):
            results[(self.quantiles[i] < images) & (images <= self.quantiles[i+1])] = BDD_IDS[i]
        return results
    
    def save(self, file_path):
        with open(file_path, "wb") as file:
            pickle.dump(self, file)
    
    @staticmethod
    def load(file_path):
        model = None
        with open(file_path, "rb") as file:
            model = pickle.load(file)
        return model
    
def iou(mask_a, mask_b, class_id):
    return np.nan_to_num(((mask_a == class_id) & (mask_b == class_id)).sum(axis=(1,2)) / ((mask_a == class_id) | (mask_b == class_id)).sum(axis=(1,2)), 0, 0, 0)

def score_model(model, x_data, mask_data, n_classes):
    results = model.predict(x_data)
    return np.array([iou(results, mask_data, i) for i in BDD_IDS]).T, results

def make_datasets(data_table, n_classes):
    n_samples = len(data_table.data)
    n_classes = len(BDD_CLASSES)
    height = data_table.data[0][1]._image.height
    width = data_table.data[0][1]._image.width

    train_data = np.array([np.array(data_table.data[i][1]._image).reshape(height, width, 3) for i in range(n_samples)])
    mask_data = np.array([np.array(data_table.data[i][3]._image).reshape(height, width) for i in range(n_samples)])
    return train_data, mask_data
