# Solar Panel Detection using Mask R-CNN

This project uses a Mask R-CNN model to detect solar panels in satellite images. You can train the model on your own COCO-format dataset and use the trained weights for inference.

## Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. **Create directories:**
   ```bash
   mkdir images results
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install git+https://github.com/akTwelve/Mask_RCNN.git
   pip install pycocotools
   ```
   > **Note:** Use `numpy<2` for best compatibility with Mask R-CNN and scientific libraries.

4. **Download pre-trained weights (optional):**
   The model uses pre-trained weights from the COCO dataset. When you run the script for the first time, the weights will be downloaded to the `model/` directory if not present.

## Dataset Structure

Your dataset should be in COCO format, with the following structure:

```
Dataset/
  train/
    image1.jpg
    ...
    _annotations.coco.json
  valid/
    imageX.jpg
    ...
    _annotations.coco.json
  test/
    ...
    _annotations.coco.json
```

- Each split (train/valid/test) has its own images and annotation file.

## Training

1. **Edit `train.py` if needed** to point to your dataset locations.
2. **Run training:**
   ```bash
   python train.py
   ```
   - Trained weights will be saved in the `logs/` directory.

## Inference (Detection)

1. **Run the detection script:**
   ```bash
   python main.py --image_path <path_to_your_image>
   ```
   - By default, this will use the latest trained weights from `logs/`.
   - To use a specific weights file:
     ```bash
     python main.py --image_path <path_to_your_image> --weights <path_to_weights.h5>
     ```
2. **View the results:**
   - The output image with detected solar panels will be saved in the `results` directory.

## Project Structure
- `main.py`: Run solar panel detection (inference) on images.
- `train.py`: Train Mask R-CNN on your COCO-format solar panel dataset.
- `model/solar_panel.py`: Custom dataset loader and config for solar panel detection.
- `model/model.py`: Inference utilities.
- `requirements.txt`: List of dependencies.
- `README.md`: This file.
- `images/`: Directory to store input images (for quick tests).
- `results/`: Directory to save output images with detected masks.
- `Dataset/`: Your COCO-format dataset (train/valid/test splits).
- `logs/`: Training logs and saved weights.

## Notes
- Make sure all masks are returned as `np.uint8` in your dataset loader to avoid dtype errors.
- If you encounter errors with `np.bool`, replace it with `bool` in the codebase.
- For best results, use Python 3.10, TensorFlow 2.10, Keras 2.10, and `numpy<2`. 
