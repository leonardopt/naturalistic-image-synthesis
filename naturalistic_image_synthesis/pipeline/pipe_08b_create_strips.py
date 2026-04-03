"""
Step 8b — Create 1×10 visual inspection strips for the final stimulus set.

For each unique object ID in the final stimulus directory, arranges its 10
images in a horizontal strip at 200×200 px per cell with filenames annotated,
and saves one JPEG per object to stimulus_set_from_interpolations_*_strips/.
"""
import math
import os

from PIL import Image, ImageDraw, ImageFont
from naturalistic_image_synthesis.config import STIM_SET_FROM_INTERPOL_DIR


def create_image_grid(source_folder, target_folder, required_string, image_size=(200, 200), font_size=20,
                      single_strip=False, str_name=''):
    """Arrange matching images into a grid or strip and save as JPEG.

    Args:
        source_folder: Directory containing the source PNGs.
        target_folder: Directory where the output JPEG is saved.
        required_string: Only files whose name contains this string are included.
        image_size: (width, height) of each cell in pixels.
        font_size: Font size for filename annotations (currently uses default font).
        single_strip: If True, force a single row regardless of image count.
        str_name: Suffix appended to the output filename before the extension.
    """
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # List all image files in the source folder that contain the required string
    image_files = [f for f in os.listdir(source_folder) if f.endswith('.png') and required_string in f]
    image_files.sort()  # Sort the files for consistent ordering

    # Determine grid size based on the number of images
    num_images = len(image_files)
    if num_images == 10 and not single_strip:
        grid_cols = 5
        grid_rows = 2
    elif single_strip:
        grid_cols = num_images
        grid_rows = 1
    else:
        grid_cols = int(math.ceil(math.sqrt(num_images)))
        grid_rows = int(math.ceil(num_images / grid_cols))

    # Create a new blank image for the grid
    grid_width = grid_cols * image_size[0]
    grid_height = grid_rows * image_size[1]
    grid_image = Image.new('RGB', (grid_width, grid_height))

    font = ImageFont.load_default()

    # Paste each image into the grid and add the filename
    for index, image_file in enumerate(image_files):
        path = os.path.join(source_folder, image_file)
        try:
            with Image.open(path) as img:
                # Resize image to fit the grid cell
                img = img.resize(image_size)
                # Calculate position for pasting
                x_index = index % grid_cols
                y_index = index // grid_cols
                x = x_index * image_size[0]
                y = y_index * image_size[1]
                # Paste the image
                grid_image.paste(img, (x, y))
                # Draw the filename
                draw = ImageDraw.Draw(grid_image)
                # Truncate the filename if it's too long to fit in the image
                truncated_filename = (image_file[:25] + '..') if len(image_file) > 25 else image_file
                draw.text((x + 5, y + 5), truncated_filename, fill="white", font=font)
        except Exception as e:
            print(f"Error processing {image_file}: {e}")

    # Save the grid image
    grid_image_path = os.path.join(target_folder, f'{required_string}_image_grid{str_name}.jpg')
    grid_image.save(grid_image_path)
    print(f"Grid image saved to {grid_image_path}")


# Specify the directories
SRCDIR = STIM_SET_FROM_INTERPOL_DIR
DST_DIR = f'{STIM_SET_FROM_INTERPOL_DIR}_strips'
required_string = 'LAN-MOT'  # Adjust this as necessary based on your filtering needs

# Ensure the destination directory exists
os.makedirs(DST_DIR, exist_ok=True)

IDs = set([file[:7] for file in os.listdir(SRCDIR) if file.endswith('.png')])
if not IDs:
    raise ValueError(f'No PNG files found in {SRCDIR}')

for id in IDs:
    # Run the grid creation process
    create_image_grid(SRCDIR, DST_DIR, required_string=id, image_size=(200, 200), font_size=20, single_strip=True,
                      str_name='strip')
