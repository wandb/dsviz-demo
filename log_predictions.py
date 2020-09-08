import argparse
import random
import numpy as np
import json
import os
from six.moves import queue
import sys
import time
from tqdm import tqdm
from PIL import Image as PILImage

import wandb
from wandb.sdk import wandb_artifacts

parser = argparse.ArgumentParser()
parser.add_argument('name', type=str)
parser.add_argument('--image_count', type=int, default=100)

BDD_CLASSES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
    'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
    'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'void'
]

def main(argv):
    args = parser.parse_args()

    run = wandb.init(job_type='predict')
    art = wandb.Artifact(args.name, type='dataset')

    bdd_seg_dir = 'bdd100k/seg'

    image_train_dir = os.path.join(bdd_seg_dir, 'images/train')

    class_set = wandb_artifacts.ClassSet(
        [{'name': c, 'id': i} for i, c in enumerate(BDD_CLASSES)])
    art.add(class_set, 'classes.json')

    full_bdd_seg_dir = os.path.join(bdd_seg_dir, 'labels', 'train')

    # Log the dataset table. (This is temporary, we're adding the ability to
    # join predictions to a dataset table that was logged in a different artifact)
    table = wandb_artifacts.Table(['example'])
    for fname in os.listdir(image_train_dir)[:args.image_count]:
        im_path = os.path.join(image_train_dir, fname)
        mask_fname = fname.split('.')[0] + '_train_id.png'
        mask_path = os.path.join(bdd_seg_dir, 'labels', 'train', mask_fname)

        class_ids = np.unique(np.array(PILImage.open(mask_path)))

        mask_img = wandb_artifacts.Image(im_path,
            masks={
                "ground_truth": {
                    "path": mask_path,
                },
            },
            classes={
                'path': 'classes.json'
            },
            present_classes=class_ids.tolist()
        )
        table.add_data(mask_img)

    art.add(table, 'examples.table.json')

    all_mask_paths = os.listdir(full_bdd_seg_dir)

    # Log the predections table.
    table = wandb_artifacts.Table(['preds', 'mean_iou'])
    for fname in os.listdir(image_train_dir)[:args.image_count]:
        im_path = os.path.join(image_train_dir, fname)
        mask_fname = fname.split('.')[0] + '_train_id.png'

        # randomly log the mask that matches ground truth, or a random mask
        mask_path = os.path.join(bdd_seg_dir, 'labels', 'train', mask_fname)
        if random.random() > 0.5:
            mask_path = os.path.join(full_bdd_seg_dir, random.choice(all_mask_paths))

        mask_img = wandb_artifacts.Image(im_path,
            masks={
                "pred": {
                    "path": mask_path
                }
            },
            classes={
                'path': 'classes.json'
            }
        )
        table.add_data(mask_img, random.random())

    art.add(table, 'preds.table.json')

    # Log a joined table that tells us to join preds to examples.
    art.add(wandb_artifacts.JoinedTable(
        'examples.table.json', 'preds.table.json', 'path'), 'preds.joined-table.json')

    # table = wandb_artifacts.Table(['a', 'b', 'c'])
    # table.add_data(1, 'hello2', 3)
    # table.add_data(4, 5, 6)
    # art.add(table, 'basic.table.json')

    run.log_artifact(art)

if __name__ == '__main__':
    main(sys.argv)
