import os
import json
import time
import math
import numpy as np

class DataPreparation:
    def __init__(self, processed_data_folder, ml_data_folder):
        self.processed_data_folder = processed_data_folder
        self.ml_data_folder = ml_data_folder
        os.makedirs(self.ml_data_folder, exist_ok=True)
        
        # Configuration for features
        self.feature_config = {
            'mouth': {
                'distances': [(13, 14), (81, 178), (311, 402), (78, 308)],
                'angles': [(13, 78, 308)]
            },
            'right eye': {
                'distances': [(159, 145), (157,154), (161,163)],
                'angles': [(33, 159, 133)]
            },
            'left eye': {
                'distances': [(386,374), (384,381), (388,390)],
                'angles': [(362,386,263)]
            },
            'right eyebrow': {
                'distances': [(52,46), (52,55)],
                'angles': [(46,52,55)]
            },
            'left eyebrow': {
                'distances': [(285,282), (282,276)],
                'angles': [(285,282,276)]
            },
            'Nose': {
                'distances': [],
                'angles': [(278,48,5)]
            }, 
            'right cheek': {
                'distances': [(101,216)],
                'angles': [(101,203,216)]
            }, 
            'left cheek': {
                'distances': [(330,436)],
                'angles': [(330,423,436)]
            },          
            # Add more features here
        }

    def find_landmark(self, data, landmark_id):
        for point in data:
            if point[0] == landmark_id:
                return point[1], point[2]
        return None, None

    def calculate_angle(self, point1, point2, point3):
        a = np.array(point1)
        b = np.array(point2)
        c = np.array(point3)
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        
        # Handle the special case where the angle is 0
        if np.isclose(cosine_angle, 1.0) or np.isclose(cosine_angle, -1.0):
            return 0.0
        
        angle = np.arccos(cosine_angle)
        return np.degrees(angle)

    def extract_features(self, data):
        features = {}
        for feature_name, feature_data in self.feature_config.items():
            print(f"Processing feature: {feature_name}")
            features[feature_name] = {'distances': {}, 'angles': {}}
            
            for dist_pair in feature_data.get('distances', []):
                x1, y1 = self.find_landmark(data, dist_pair[0])
                x2, y2 = self.find_landmark(data, dist_pair[1])
                
                if x1 is not None and x2 is not None:
                    print(f"Distance pair: {dist_pair[0]} and {dist_pair[1]}")
                    print(f"Point1 (ID {dist_pair[0]}): x={x1}, y={y1}")
                    print(f"Point2 (ID {dist_pair[1]}): x={x2}, y={y2}")
                    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    features[feature_name]['distances'][f'distance_{dist_pair[0]}_{dist_pair[1]}'] = distance
                    print(f"Calculated distance: {distance}")
                    time.sleep(1)
                else:
                    print(f"Warning: Landmark IDs {dist_pair[0]} or {dist_pair[1]} are not found in the data.")
            
            for angle_triplet in feature_data.get('angles', []):
                x1, y1 = self.find_landmark(data, angle_triplet[0])
                x2, y2 = self.find_landmark(data, angle_triplet[1])
                x3, y3 = self.find_landmark(data, angle_triplet[2])
                
                if x1 is not None and x2 is not None and x3 is not None:
                    print(f"Angle triplet: {angle_triplet[0]}, {angle_triplet[1]}, and {angle_triplet[2]}")
                    print(f"Point1 (ID {angle_triplet[0]}): x={x1}, y={y1}")
                    print(f"Point2 (ID {angle_triplet[1]}): x={x2}, y={y2}")
                    print(f"Point3 (ID {angle_triplet[2]}): x={x3}, y={y3}")
                    angle = self.calculate_angle((x1, y1), (x2, y2), (x3, y3))
                    features[feature_name]['angles'][f'angle_{angle_triplet[0]}_{angle_triplet[1]}_{angle_triplet[2]}'] = angle
                    print(f"Calculated angle: {angle}")
                else:
                    print(f"Warning: Landmark IDs {angle_triplet[0]}, {angle_triplet[1]}, or {angle_triplet[2]} are not found in the data.")
        
        return features

    def prepare_ml_data(self):
        for root, dirs, files in os.walk(self.processed_data_folder):
            for file in files:
                if file.startswith('face') and file.endswith('data.json'):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r') as f:
                        data = json.load(f)[0]  # Assuming the data is wrapped in an extra list
                    
                    print(f"Processing file: {file_path}")
                    print(f"Data length: {len(data)}")
                    
                    features = self.extract_features(data)
                    
                    # Save the features to the ml_data folder
                    relative_path = os.path.relpath(file_path, self.processed_data_folder)
                    save_folder = os.path.join(self.ml_data_folder, os.path.dirname(relative_path))
                    os.makedirs(save_folder, exist_ok=True)
                    save_path = os.path.join(save_folder, file.replace('data.json', 'features.json'))
                    with open(save_path, 'w') as f:
                        json.dump(features, f)
                    print(f"Saved features to: {save_path}")

if __name__ == "__main__":
    processed_data_folder = 'processed_data'
    ml_data_folder = 'ml_data'
    data_preparation = DataPreparation(processed_data_folder, ml_data_folder)
    data_preparation.prepare_ml_data()
    print("Data preparation complete.")