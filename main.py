import argparse
import os
from model.model import get_model, detect_solar_panels, save_result_image

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Detect solar panels in an image.')
    parser.add_argument('--image_path', required=False, help='Path to the image file')
    parser.add_argument('--weights', required=False, help='Path to trained weights (.h5) file. If not provided, will use latest in logs/.')
    args = parser.parse_args()

    if not args.image_path:
        print("Please provide the path to a satellite image of solar panels using the --image_path argument.")
        return

    # Create results directory if it doesn't exist
    if not os.path.exists("results"):
        os.makedirs("results")

    # Load the model
    model = get_model(weights_path=args.weights)

    # Detect solar panels
    result, image = detect_solar_panels(model, args.image_path)

    # Save the output
    file_name = os.path.basename(args.image_path)
    save_path = os.path.join("results", file_name)
    save_result_image(image, result, save_path)
    print(f"Result saved to {save_path}")

if __name__ == '__main__':
    main() 
