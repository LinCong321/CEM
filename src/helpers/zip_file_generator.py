import os
import re
import zipfile
import pandas as pd
from tqdm import tqdm

def clean_title(title):
    """Clean the title to remove illegal characters for filenames."""
    # Define a regular expression for illegal characters in filenames
    illegal_chars = r'[<>:"/\\|?*]'
    # Replace illegal characters with an underscore or just remove them
    return re.sub(illegal_chars, '_', title)

def create_output_dirs(base_dir, num_files_per_folder):
    """Create the base output directory and subdirectories."""
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    folder_num = 0
    folder_path = os.path.join(base_dir, f"{folder_num:03d}")
    os.makedirs(folder_path, exist_ok=True)
    
    return base_dir, folder_path, folder_num

def process_parquet_files(parquet_dir, output_dir, num_files_per_folder=10000):
    """Process multiple Parquet files, extract text and save to txt, and zip them."""
    # Prepare to create directories and zip files
    base_output_dir, current_folder_path, folder_num = create_output_dirs(output_dir, num_files_per_folder)
    
    txt_counter = 0
    zip_counter = 0
    
    # Assuming the parquet files are named in a sequential pattern
    parquet_files = [os.path.join(parquet_dir, f'train-{str(i).zfill(5)}-of-00006.parquet') for i in range(6)]

    # Loop through all parquet files
    for parquet_file in parquet_files:
        # Load the parquet file
        df = pd.read_parquet(parquet_file)

        # Wrap the DataFrame iteration with tqdm to show progress
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {parquet_file}", unit="record"):
            title = row['title']
            text = row['text']

            # Clean the title to remove illegal characters
            clean_filename = clean_title(title)

            # Define txt file path
            txt_filename = f"{clean_filename}.txt"
            txt_file_path = os.path.join(current_folder_path, txt_filename)

            # Write the text to a .txt file
            with open(txt_file_path, 'w', encoding='utf-8') as f:
                f.write(text)

            # Increment counter for txt files
            txt_counter += 1

            # If we have reached the max number of files, start a new folder
            if txt_counter >= num_files_per_folder:
                folder_num += 1
                current_folder_path = os.path.join(base_output_dir, f"{folder_num:03d}")
                os.makedirs(current_folder_path, exist_ok=True)
                txt_counter = 0  # Reset the counter for the new folder

            # Now zip the file
            zip_filename = f"{clean_filename}.zip"
            zip_file_path = os.path.join(current_folder_path, zip_filename)

            with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                zipf.write(txt_file_path, arcname=txt_filename)

            # Remove the original txt file after zipping (optional)
            os.remove(txt_file_path)

            # Increment zip file counter
            zip_counter += 1

    print(f"Finished processing {zip_counter} zip files.")

# Example usage
def generate():
    parquet_file = './parquet'
    output_dir = './output'
    process_parquet_files(parquet_file, output_dir)
