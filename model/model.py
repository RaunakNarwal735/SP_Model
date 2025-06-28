import os
import numpy as np
import skimage.io
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.config import Config
import matplotlib.pyplot as plt
import tensorflow as tf
import re
tf.compat.v1.enable_eager_execution()

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join("model", "mask_rcnn_coco.h5")

class InferenceConfig(Config):

    """
    Configuration for the Mask R-CNN model used during inference.
    """
    NAME = "solar"

    # Number of classes (background + solar panel)
    NUM_CLASSES = 1 + 1

    # Detection confidence threshold
    DETECTION_MIN_CONFIDENCE = 0.9
 # Modify if your model was trained with different size

    # Runtime settings
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


def get_model(weights_path=None):
    """
    Loads the Mask R-CNN model for inference. If weights_path is provided, loads those weights; otherwise, loads the latest trained weights from logs/ if available, else COCO weights.
    """
    config = InferenceConfig()
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir="logs/")

    # If a custom weights path is provided, use it
    if weights_path and os.path.exists(weights_path):
        print(f"Loading weights from {weights_path}")
        model.load_weights(weights_path, by_name=True)
        return model

    # Try to find the latest trained weights in logs/
    logs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
    if os.path.exists(logs_dir):
        weight_files = []
        for root, dirs, files in os.walk(logs_dir):
            for file in files:
                if file.endswith('.h5') and 'mask_rcnn_solar_panel' in file:
                    weight_files.append(os.path.join(root, file))
        if weight_files:
            latest_weights = max(weight_files, key=os.path.getctime)
            print(f"Loading latest trained weights: {latest_weights}")
            model.load_weights(latest_weights, by_name=True)
            return model

    # Fallback to COCO weights
    print(f"Loading COCO weights from {COCO_WEIGHTS_PATH}")
    if not os.path.exists(COCO_WEIGHTS_PATH):
        utils.download_trained_weights(COCO_WEIGHTS_PATH)
    model.load_weights(COCO_WEIGHTS_PATH, by_name=True, exclude=[
        "mrcnn_class_logits", "mrcnn_bbox_fc",
        "mrcnn_bbox", "mrcnn_mask"
    ])
    return model


def detect_solar_panels(model, image_path):
    """
    Runs detection on a single image and returns the results.
    """
    image = skimage.io.imread(image_path)
    results = model.detect([image], verbose=1)
    r = results[0]
    return r, image


def save_result_image(image, result, save_path):
    """
    Saves the image with detected solar panel masks to a file.
    """
    visualize.display_instances(
        image,
        result['rois'],
        result['masks'],
        result['class_ids'],
        ['BG', 'solar panel'],
        result['scores'],
        title="Solar Panels"
    )
    plt.savefig(save_path)
    plt.close()  # Optional: closes the figure to free memory

def get_initial_epoch_from_weights(weights_path):
    # Extract epoch number from filename, e.g., mask_rcnn_solar_panel_0005.h5
    match = re.search(r"_(\d{4})\\.h5$", weights_path)
    if match:
        return int(match.group(1))
    return 0
