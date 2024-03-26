import os
import pandas as pd
import requests
from PIL import Image
from io import BytesIO

# Load the CSV file from Google's cloud storage.
# This example assumes you are using the train dataset.
metadata = pd.read_csv('https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv')

# Randomly select 500000 images
metadata = metadata.sample(n=5)

# Function to download an image from a URL
def download_image(url, image_file_path):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    gray_image = image.convert('L')  # convert image to grayscale
    resized_image = gray_image.resize((300, 300))  # resize image to 300x300
    resized_image.save(image_file_path, 'PNG')

# Create a directory to store the images
os.makedirs('images', exist_ok=True)

# Prepare an empty DataFrame to store the downloaded images' data
downloaded_images = pd.DataFrame(columns = ['ImageID', 'OriginalURL'])

# Iterate over the rows of the DataFrame, downloading each image.
for i, row in metadata.iterrows():
    image_file_path = os.path.join('images', f"{row['ImageID']}.png")  # images will be saved as .png
    try:
        download_image(row['OriginalURL'], image_file_path)
        downloaded_images = downloaded_images.append(row)
        print(f"Downloaded {i+1} of 500000 images")
    except:
        print(f"Failed to download {i+1} of 500000 images")

print("Download complete")

# Save the DataFrame to a CSV file
downloaded_images.to_csv('downloaded_images.csv', index=False)
