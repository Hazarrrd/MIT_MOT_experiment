from PIL import Image
import os

# Folder containing images
input_folder = "/home/janek/psychologia/MIT_MOT_experiment/icons_mit/icons_snowflakes"
output_folder = "/home/janek/psychologia/MIT_MOT_experiment/icons_mit/crops"

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Process all images in the folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        img_path = os.path.join(input_folder, filename)
        img = Image.open(img_path)

        # Ensure image is 1000x1000 before cropping
        if img.size == (1000, 1000):
            left = (1000 - 500) // 2  # 250
            top = (1000 - 500) // 2   # 250
            right = left + 500        # 750
            bottom = top + 500        # 750

            cropped_img = img.crop((left, top, right, bottom))
            cropped_img.save(os.path.join(output_folder, filename))

print("Cropping completed!")