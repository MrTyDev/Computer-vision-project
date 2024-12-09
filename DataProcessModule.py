import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import facemeshModul as fm
import json
import cv2

class TrainingModule:
    def __init__(self, environmentpath=os.getcwd(), testfolderpath=os.path.join(os.getcwd(), "test"), trainfolderpath=os.path.join(os.getcwd(), "train")):
        self.testfolderpath = testfolderpath
        self.trainfolderpath = trainfolderpath
        self.processed_data_folder = os.path.join(environmentpath, "processed_data")
        os.makedirs(self.processed_data_folder, exist_ok=True)
        self.face_mesh_module = fm.faceMeshModule()

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

    def process_images(self, FolderPath):
        for root, dirs, files in os.walk(FolderPath):
            for idx, file in enumerate(files):
                if file.endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(root, file)
                    img = cv2.imread(image_path)
                    if img is not None:
                        try:
                            result = self.face_mesh_module.findFaceMesh(img, draw=False)
                            if result is not None:
                                _, faces = result
                                if faces:
                                    relative_path = os.path.relpath(image_path, FolderPath)
                                    save_folder = os.path.join(self.processed_data_folder, os.path.dirname(relative_path))
                                    os.makedirs(save_folder, exist_ok=True)
                                    save_path = os.path.join(save_folder, f"face{idx+1}data.json")
                                    with open(save_path, 'w') as f:
                                        json.dump(faces, f)
                                    print(f"Processed and saved: {save_path}")
                                else:
                                    print(f"No face detected in image: {image_path}")
                            else:
                                print(f"Face mesh detection failed for: {image_path}")
                        except Exception as e:
                            print(f"Error processing image {image_path}: {str(e)}")
                    else:
                        print(f"Error loading image: {image_path}")

if __name__ == "__main__":
    training_module = TrainingModule()
    training_module.process_images(training_module.trainfolderpath)
    print("Processing complete.")