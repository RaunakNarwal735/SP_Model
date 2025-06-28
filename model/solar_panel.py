import os
from mrcnn.config import Config
from mrcnn import utils
import json
import skimage.io
import numpy as np
from pycocotools import mask as maskUtils

class SolarPanelConfig(Config):
    NAME = "solar_panel"
    NUM_CLASSES = 1 + 1  # background + solar panel
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    STEPS_PER_EPOCH = 1000
    VALIDATION_STEPS = 50
    DETECTION_MIN_CONFIDENCE = 0.7
    # Adjust image size if needed
    IMAGE_MIN_DIM = 640
    IMAGE_MAX_DIM = 640
    BACKBONE = "resnet50"

class SolarPanelCocoDataset(utils.Dataset):
    def load_coco(self, annotation_json, images_dir, class_names=['solar panel']):
        # Add classes
        for i, name in enumerate(class_names):
            self.add_class("solar_panel", i+1, name)
        # Load annotations
        with open(annotation_json) as f:
            coco = json.load(f)
        # Map image IDs to file names
        image_id_to_filename = {img['id']: img['file_name'] for img in coco['images']}
        # Map image IDs to annotations
        image_id_to_anns = {}
        for ann in coco['annotations']:
            image_id_to_anns.setdefault(ann['image_id'], []).append(ann)
        # Add images
        for image_id, file_name in image_id_to_filename.items():
            anns = image_id_to_anns.get(image_id, [])
            self.add_image(
                "solar_panel",
                image_id=image_id,
                path=os.path.join(images_dir, file_name),
                annotations=anns,
                width=coco['images'][image_id-1]['width'],
                height=coco['images'][image_id-1]['height']
            )

    def load_mask(self, image_id):
        image_info = self.image_info[image_id]
        annotations = image_info['annotations']
        height = image_info['height']
        width = image_info['width']
        masks = []
        class_ids = []
        for ann in annotations:
            if 'segmentation' in ann and ann['segmentation']:
                # Handle both polygon and RLE
                if isinstance(ann['segmentation'], list):
                    # Polygon
                    rles = maskUtils.frPyObjects(ann['segmentation'], height, width)
                    rle = maskUtils.merge(rles)
                else:
                    # RLE
                    rle = ann['segmentation']
                mask = maskUtils.decode(rle)
                masks.append(mask)
                class_ids.append(1)  # Only one class: solar panel
        if masks:
            mask = np.stack(masks, axis=-1)
            mask = mask.astype(np.uint8)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            return np.empty((height, width, 0), dtype=np.uint8), np.array([], dtype=np.int32)
