import kagglehub
import shutil
import os

# 1. Download/Update from Kaggle
print("Downloading latest NASA dataset...")
cache_path = kagglehub.dataset_download("behrad3d/nasa-cmaps")

target_dir = 'data'

# 2. Search for the 'CMaps' folder specifically
print("Finding the data source...")
for root, dirs, files in os.walk(cache_path):
    # If 'CMaps folder is found label its path as "source_folder"    
    if 'CMaps' in dirs:
        source_folder = os.path.join(root, 'CMaps')
        
        # 3. Import the folder contents to your project
        # Check if the target directory already exists
        if os.path.exists(target_dir):
            # If the directory exists delete the folder and all subdirectories
            shutil.rmtree(target_dir)

        # Create a new directory, importing the files from "source_folder"
        shutil.copytree(source_folder, target_dir)

        # Re insert the ".gitkeep" file to maintain folder structure on GitHub
        with open(os.path.join(target_dir, '.gitkeep'), 'w') as f:
            pass

        print(f"--- SUCCESS: Imported data from {source_folder} to /{target_dir} ---")
        break

# List what we got
print("Files in your project data folder:", os.listdir('data'))