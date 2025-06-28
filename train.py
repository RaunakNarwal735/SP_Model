import os
from model.solar_panel import SolarPanelConfig, SolarPanelCocoDataset
import mrcnn.model as modellib
from pycocotools import mask as maskUtils
import numpy as np
import glob
import re

# Paths
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
TRAIN_IMAGES = os.path.join(ROOT_DIR, 'Dataset', 'train')
TRAIN_ANN = os.path.join(TRAIN_IMAGES, '_annotations.coco.json')
VAL_IMAGES = os.path.join(ROOT_DIR, 'Dataset', 'valid')
VAL_ANN = os.path.join(VAL_IMAGES, '_annotations.coco.json')
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, 'model', 'mask_rcnn_coco.h5')
LOGS_DIR = os.path.join(ROOT_DIR, 'logs')

# Config
config = SolarPanelConfig()
config.display()

# Datasets
train_dataset = SolarPanelCocoDataset()
train_dataset.load_coco(TRAIN_ANN, TRAIN_IMAGES)
train_dataset.prepare()

val_dataset = SolarPanelCocoDataset()
val_dataset.load_coco(VAL_ANN, VAL_IMAGES)
val_dataset.prepare()

def find_latest_weights(logs_dir):
    weight_files = glob.glob(os.path.join(logs_dir, '**', 'mask_rcnn_solar_panel_*.h5'), recursive=True)
    if weight_files:
        return max(weight_files, key=os.path.getctime)
    return None

def get_initial_epoch_from_weights(weights_path):
    # Extract epoch number from filename, e.g., mask_rcnn_solar_panel_0005.h5
    match = re.search(r"_(\d{4})\.h5$", weights_path)
    if match:
        return int(match.group(1))
    return 0

# Model
model = modellib.MaskRCNN(mode="training", config=config, model_dir=LOGS_DIR)

# Load latest weights if available, else COCO weights
latest_weights = find_latest_weights(LOGS_DIR)
initial_epoch = 0
if latest_weights:
    print(f"Resuming training from latest checkpoint: {latest_weights}")
    model.load_weights(latest_weights, by_name=True)
    initial_epoch = get_initial_epoch_from_weights(latest_weights)
    print(f"Resuming from epoch {initial_epoch + 1}")
else:
    print(f"Starting training from COCO weights: {COCO_WEIGHTS_PATH}")
    model.load_weights(COCO_WEIGHTS_PATH, by_name=True, exclude=[
        "mrcnn_class_logits", "mrcnn_bbox_fc",
        "mrcnn_bbox", "mrcnn_mask"
    ])

# Train
model.train(
    train_dataset, val_dataset,
    learning_rate=config.LEARNING_RATE,
    epochs=20,
    layers='heads',
    initial_epoch=1
) 
