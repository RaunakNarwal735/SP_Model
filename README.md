# Solar Panel Detection using Mask R-CNN

This project uses a Mask R-CNN model to detect solar panels in satellite images.

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
   ```

4. **Download pre-trained weights:**
   The model uses pre-trained weights from the COCO dataset. When you run the script for the first time, the weights will be downloaded to the `model/` directory.

## Usage

1. **Place your satellite images** in the `images` directory.
2. **Run the detection script:**
   ```bash
   python main.py --image_path <path_to_your_image>
   ```
3. **View the results:**
   The output image with detected solar panels will be saved in the `results` directory.

## Project Structure
- `main.py`: The main script to run the solar panel detection.
- `requirements.txt`: A list of the dependencies.
- `README.md`: This file.
- `model/`: This will contain the model related files.
- `images/`: Directory to store input images.
- `results/`: Directory to save the output images with detected masks. 