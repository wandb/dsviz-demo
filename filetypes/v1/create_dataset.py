import argparse
import random
import numpy as np
import json
import os
from six.moves import queue
import shutil
import sys
import threading
import tempfile
import time
from tqdm import tqdm
from PIL import Image as PILImage

import wandb
from wandb.compat import tempfile
from wandb.sdk import wandb_artifacts

parser = argparse.ArgumentParser()
parser.add_argument('name', type=str)
parser.add_argument('--image_count', type=int, default=105)
    
BDD_CLASSES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
    'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
    'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'void'
]

def main(argv):
    args = parser.parse_args()

    run = wandb.init(job_type='create_dataset')
    art = wandb.Artifact(args.name, type='dataset')

    bdd_seg_dir = 'bdd100k/seg'

    image_train_dir = os.path.join(bdd_seg_dir, 'images/train')

    class_set = wandb_artifacts.ClassSet(
        [{'name': c, 'id': i} for i, c in enumerate(BDD_CLASSES)])
    art.add(class_set, 'classes.json')

    # table = wandb_artifacts.Table(['example', 'present_classes'])
    table = wandb_artifacts.Table(['example'])
    for fname in os.listdir(image_train_dir)[:args.image_count]:
        im_path = os.path.join(image_train_dir, fname)
        mask_fname = fname.split('.')[0] + '_train_id.png'
        mask_path = os.path.join(bdd_seg_dir, 'labels', 'train', mask_fname)

        class_ids = np.unique(np.array(PILImage.open(mask_path))).tolist()

        mask_img = wandb_artifacts.Image(im_path,
            masks={
                "ground_truth": {
                    "path": mask_path,
                },
            },
            classes={
                'path': 'classes.json'
            },
        )
        # TODO: use a media object to represent class_ids so we have
        #     a type to hint to the UI to render the correct thing.
        # note, we don't need to do this if present classes is feature
        #     of the example itself, since we know the type of wandb.Image
        # table.add_data(mask_img, class_ids)
        table.add_data(mask_img)

    art.add(table, 'examples.table.json')

    run.log_artifact(art)

if __name__ == '__main__':
    main(sys.argv)
