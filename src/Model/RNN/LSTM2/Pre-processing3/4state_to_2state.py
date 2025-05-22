import os
import json
import shutil

# Base directory
base_dir = r"src\Model\RNN\LSTM2\Pre-processing3"

# Get all folders that match the pattern "processed_data_*ms"
source_folders = [f for f in os.listdir(base_dir) if f.startswith("processed_data_") and f.endswith("ms")]

for folder in source_folders:
    # Create the new folder name with "_2state" suffix
    new_folder = folder + "_2state"
    new_folder_path = os.path.join(base_dir, new_folder)
    
    # Create the new folder if it doesn't exist
    os.makedirs(new_folder_path, exist_ok=True)
    
    # Path to source json file
    source_json = os.path.join(base_dir, folder, "processed_data.json")
    
    # Path to destination json file
    dest_json = os.path.join(new_folder_path, "processed_data.json")
    
    # Read the source json file
    with open(source_json, 'r') as f:
        data = json.load(f)
    
    # Convert labels: 0->0, {1,2,3}->1
    for item in data:
        if 'label' in item:
            item['label'] = 0 if item['label'] == 0 else 1
    
    # Save the modified data to the new location
    with open(dest_json, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"Processed {folder} -> {new_folder}")

print("All folders have been processed!")