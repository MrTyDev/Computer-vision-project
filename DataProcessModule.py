import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import facemeshModul as fm
import json
import cv2
import math

class DataProcessing:
    def __init__(self, environmentpath=os.getcwd(), testfolderpath=os.path.join(os.getcwd(), "test"), trainfolderpath=os.path.join(os.getcwd(), "train")):
        self.testfolderpath = testfolderpath
        self.trainfolderpath = trainfolderpath
        self.processed_data_folder = os.path.join(environmentpath, "processed_data")
        os.makedirs(self.processed_data_folder, exist_ok=True)
        self.face_mesh_module = fm.faceMeshModule()

    def calculate_distance(self, point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        distance = math.hypot(x2 - x1, y2 - y1)
        return distance  

    def calculate_angle(self, pointA, pointB, pointC):
        a = np.array(pointA)
        b = np.array(pointB)
        c = np.array(pointC)
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)
        return np.degrees(angle)

    def process_images(self, FolderPath, num_images_per_folder):
        for root, dirs, files in os.walk(FolderPath):
            if not files:
                continue
            image_files = [file for file in files if file.endswith(('.jpg', '.jpeg', '.png'))]
            # Limit the number of images to process
            images_to_process = image_files[:num_images_per_folder]
            for idx, file in enumerate(images_to_process):
                image_path = os.path.join(root, file)
                img = cv2.imread(image_path)
                if img is not None:
                    try:
                        result = self.face_mesh_module.findFaceMesh(img, draw=False)
                        if result is not None:
                            _, faces = result
                            if faces:
                                for face_idx, face_landmarks in enumerate(faces):
                                    landmark_coords = {}
                                    for landmark in face_landmarks:
                                        id, x, y = landmark
                                        landmark_coords[id] = (x, y)
                                    required_mouth_landmarks = [13, 14, 78, 81, 178, 311, 402]
                                    if all(lm in landmark_coords for lm in required_mouth_landmarks):
                                        # Calculate distances
                                        dist_13_14 = self.calculate_distance(landmark_coords[13], landmark_coords[14])
                                        dist_81_178 = self.calculate_distance(landmark_coords[81], landmark_coords[178])
                                        dist_311_402 = self.calculate_distance(landmark_coords[311], landmark_coords[402])
                                        # Calculate angle
                                        angle_13_78_14 = self.calculate_angle(landmark_coords[13], landmark_coords[78], landmark_coords[14])
                                        # Store calculations
                                        calculations = {
                                            'distances': {
                                                '13_14': dist_13_14,
                                                '81_178': dist_81_178,
                                                '311_402': dist_311_402
                                            },
                                            'angles': {
                                                '13_78_14': angle_13_78_14
                                            }
                                        }
                                        # Save calculations
                                        relative_path = os.path.relpath(image_path, FolderPath)
                                        save_folder = os.path.join(self.processed_data_folder, os.path.dirname(relative_path))
                                        os.makedirs(save_folder, exist_ok=True)
                                        save_path = os.path.join(save_folder, f"face{idx+1}_calculations.json")
                                        with open(save_path, 'w') as f:
                                            json.dump(calculations, f)
                                        print(f"Processed and saved calculations: {save_path}")
                                    else:
                                        print(f"Required landmarks not found in face {face_idx} of image {image_path}")
                            else:
                                print(f"No face detected in image: {image_path}")
                        else:
                            print(f"Face mesh detection failed for: {image_path}")
                    except Exception as e:
                        print(f"Error processing image {image_path}: {str(e)}")
                else:
                    print(f"Error loading image: {image_path}")

if __name__ == "__main__":
    num_images = int(input("Enter the number of images to process from each folder: "))
    Processing_module = DataProcessing()
    Processing_module.process_images(Processing_module.trainfolderpath, num_images)
    print("Processing complete.")