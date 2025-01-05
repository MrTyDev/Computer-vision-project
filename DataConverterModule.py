import os
import json
import pandas as pd
import csv

class DataConverterModule:
    def __init__(self, environmentpath=os.getcwd(), testfolderpath=os.path.join(os.getcwd(), "test"), trainfolderpath=os.path.join(os.getcwd(), "train")):
        self.testfolderpath = testfolderpath
        self.trainfolderpath = trainfolderpath
        self.ml_data_folder = os.path.join(environmentpath, "ml_data")
        self.environmentpath = environmentpath
        os.makedirs(self.ml_data_folder, exist_ok=True)

    def search_folder(self, FolderPath):
        # List all files in the folder
        files = os.listdir(FolderPath)
        return files

    def open_folder_list_files(self, FolderPath):
        # List all files in the folder
        data = []
        for root, dirs, files in os.walk(FolderPath):
            for file in files:
                if file.endswith(".jpeg") or file.endswith(".jpg"):
                    file_path = os.path.join(root, file)
                    data.append(file_path)
        return data

    def process_images(self, FolderPath, num_images_per_folder):
        for root, dirs, files in os.walk(FolderPath):
            if not files:
                continue
            image_files = [file for file in files if file.endswith(('.jpg', '.jpeg', '.png'))]
            images_to_process = image_files[:num_images_per_folder]
            for idx, file in enumerate(images_to_process):
                image_path = os.path.join(root, file)
                # Process the image (implementation not shown)
    
    def convert_json_to_csv(self, exclude_targets=[]):
        csv_file_path = os.path.join(self.environmentpath, "Training_data.csv")
        all_features = []
        data_rows = []

        for root, dirs, files in os.walk(self.ml_data_folder):
            for file in files:
                if file.endswith("features.json"):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r') as f:
                        features = json.load(f)
                        row = {}
                        for feature_name, feature_data in features.items():
                            for dist_name, dist_value in feature_data.get('distances', {}).items():
                                feature_key = f"{feature_name}_{dist_name}"
                                row[feature_key] = dist_value
                                if feature_key not in all_features:
                                    all_features.append(feature_key)
                            for angle_name, angle_value in feature_data.get('angles', {}).items():
                                feature_key = f"{feature_name}_{angle_name}"
                                row[feature_key] = angle_value
                                if feature_key not in all_features:
                                    all_features.append(feature_key)
                        # Add the target value (folder name)
                        target = os.path.basename(os.path.dirname(file_path))
                        row['target'] = target
                        if 'target' not in all_features:
                            all_features.append('target')
                        data_rows.append(row)

        # Exclude selected targets
        if exclude_targets:
            data_rows = [row for row in data_rows if row['target'] not in exclude_targets]

        # Write to CSV
        with open(csv_file_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=all_features)
            writer.writeheader()
            for row in data_rows:
                writer.writerow(row)

        print(f"CSV file created at: {csv_file_path}")

if __name__ == "__main__":
    data_converter = DataConverterModule()
    data_converter.convert_json_to_csv()
    print("Processing complete.")