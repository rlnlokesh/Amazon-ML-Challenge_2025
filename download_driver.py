# download_driver.py

import pandas as pd
import os
# Assuming your utility functions are correctly placed in src/utils.py
from utils import download_all_images

# --- Configuration ---
DATA_FILE = "D:/DA/68e8d1d70b66d_student_resource/student_resource/dataset/test.csv"
IMAGE_DIR = "images_test_all"  # Renamed directory for clarity
MAX_IMAGES = 75000  # <--- New configuration variable

def main():
    print("Starting VS Code environment execution...")
    
    if not os.path.exists(DATA_FILE):
        print(f"Error: Training data not found at {DATA_FILE}. Please check the path.")
        return

    # 1. Load the training data
    df_train = pd.read_csv(DATA_FILE)
    print(f"Loaded {len(df_train)} total training records.")

    # 2. Limit the DataFrame to the first 37,500 records
    df_subset = df_train.head(MAX_IMAGES).copy()
    print(f"Limiting download to the first {len(df_subset)} images.")

    # 3. Start the parallel download
    # num_workers: 64 is a safe starting point for parallel I/O.
    df_results = download_all_images(
        df=df_subset,  # <-- Pass the subsetted DataFrame
        download_folder=IMAGE_DIR,
        num_workers=64 
    )
    
    # 4. Save and report on failures
    df_results.to_csv("download_results_test_75000.csv", index=False)
    failed_df = df_results[~df_results['status'].isin(['Success', 'Exists', 'NoLink'])]
    print(f"\nTotal failed downloads: {len(failed_df)} out of {len(df_subset)}.")
    print(f"Final status saved to dataset/download_results_37500.csv")


if __name__ == "__main__":
    main()