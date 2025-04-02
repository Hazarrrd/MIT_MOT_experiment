import os
import cv2
import numpy as np
import argparse

def add_black_border(image_path, output_path, border_size=5):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"Skipping {image_path}, unable to read image.")
        return
    
    # Ensure image has an alpha channel
    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    
    # Draw black borders directly on the image
    image[:border_size, :, :] = [0, 0, 0, 255]  # Top border
    image[-border_size:, :, :] = [0, 0, 0, 255]  # Bottom border
    image[:, :border_size, :] = [0, 0, 0, 255]  # Left border
    image[:, -border_size:, :] = [0, 0, 0, 255]  # Right border
    
    cv2.imwrite(output_path, image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    
    # Create negative image
    negative_image = cv2.bitwise_not(image)
    negative_image[:, :, :3] = 255 - negative_image[:, :, :3]  # Invert RGB only
    # Draw black borders directly on the image
    negative_image[:border_size, :, :] = [0, 0, 0, 255]  # Top border
    negative_image[-border_size:, :, :] = [0, 0, 0, 255]  # Bottom border
    negative_image[:, :border_size, :] = [0, 0, 0, 255]  # Left border
    negative_image[:, -border_size:, :] = [0, 0, 0, 255]  # Right border
    mask = negative_image[:, :, 3] > 0  # Only modify non-transparent areas
    negative_image[mask, :3] = [0,0,0] # Invert RGB only
    negative_output_path = output_path.replace(".png", "_negative.png").replace(".jpg", "_negative.jpg")
    cv2.imwrite(negative_output_path, negative_image, [cv2.IMWRITE_PNG_COMPRESSION, 9])

def process_images(input_dir, output_dir, border_size=5):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            add_black_border(input_path, output_path, border_size)
            print(f"Processed: {filename}")
    
if __name__ == "__main__":
    input_dir = "/home/janek/psychologia/MIT_MOT_experiment/icons_mit/icons_shapes"
    output_dir = "/home/janek/psychologia/MIT_MOT_experiment/icons_mit/icons_shapes_b"
    border_size = 20
    process_images(input_dir, output_dir, border_size)
