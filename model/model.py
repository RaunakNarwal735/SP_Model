import os
import sys
import numpy as np
import skimage.io
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join("model", "mask_rcnn_coco.h5")

class InferenceConfig(object):
    """
    Configuration for the model.
    """
    # Give the configuration a recognizable name
    NAME = "solar"

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + solar panel

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

    # Images per GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


def get_model():
    """
    This function will load the pre-trained model
    """
    config = InferenceConfig()
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir="model/")

    # Download COCO weights if not available
    if not os.path.exists(COCO_WEIGHTS_PATH):
        utils.download_trained_weights(COCO_WEIGHTS_PATH)

    model.load_weights(COCO_WEIGHTS_PATH, by_name=True, exclude=[
        "mrcnn_class_logits", "mrcnn_bbox_fc",
        "mrcnn_bbox", "mrcnn_mask"])
    
    return model


def detect_solar_panels(model, image_path):
    """
    This function takes an image and returns the detected solar panels
    """
    image = skimage.io.imread(image_path)
    results = model.detect([image], verbose=1)
    r = results[0]
    return r, image


def save_result_image(image, result, save_path):
    """
    This function will save the image with detected masks
    """
    visualize.display_instances(image, result['rois'], result['masks'], result['class_ids'],
                                ['BG', 'solar panel'], result['scores'],
                                title="Solar Panels",
                                save_path=save_path) 
