import gdown
import zipfile
import os

# Function to download a file from Google Drive
def download_from_drive(drive_url, output_path):
    gdown.download(drive_url, output_path, quiet=False)

# Function to unzip a file
def unzip_file(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

# Replace this URL with the shareable link of your Google Drive file
drive_url = 'https://drive.google.com/uc?id=1gKYZAxkv07aP_5oiWyxVqQGM1-7huMXx'

# Local paths
output_path = 'data.zip'
extract_to = './'

# Download the file
download_from_drive(drive_url, output_path)

# Create the directory to extract files if it doesn't exist
if not os.path.exists(extract_to):
    os.makedirs(extract_to)

# Unzip the downloaded file
unzip_file(output_path, extract_to)

print(f"File downloaded and unzipped to '{extract_to}'")
