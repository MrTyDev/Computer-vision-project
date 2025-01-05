import tkinter as tk
from tkinter import messagebox, scrolledtext
import os
from DataConverterModule import DataConverterModule
from DataProcessModule import TrainingModule
from DataPreparationModule import DataPreparation
import joblib
import pandas as pd
import sys
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from facemeshModul import faceMeshModule
import cv2
import time
import numpy as np

class HandlerUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ML Project Handler")
        self.geometry("800x600")

        self.converter = DataConverterModule()
        self.processor = TrainingModule()
        self.preparer = DataPreparation('processed_data', 'ml_data')
        self.face_mesh = faceMeshModule()
        self.cap = None

        if os.path.exists('random_forest_model.pkl'):
            self.model = joblib.load('random_forest_model.pkl')
        else:
            self.model = None
            print("Model file not found. Please train the model first.")

        tk.Label(self, text="Number of images to process:").pack()
        self.num_images_entry = tk.Entry(self)
        self.num_images_entry.pack()

        tk.Button(self, text="Process Images", command=self.process_images).pack(pady=5)
        tk.Button(self, text="Prepare ML Data", command=self.prepare_ml_data).pack(pady=5)
        tk.Button(self, text="Convert JSON to CSV", command=self.convert_json_to_csv).pack(pady=5)

        tk.Label(self, text="Select targets to exclude:").pack()
        self.target_listbox = tk.Listbox(self, selectmode=tk.MULTIPLE)
        self.target_listbox.pack(pady=5)
        self.load_targets()

        tk.Button(self, text="Update Targets", command=self.load_targets).pack(pady=5)
        tk.Button(self, text="Train Model", command=self.train_model).pack(pady=5)

        tk.Label(self, text="Camera Index:").pack()
        self.camera_index_entry = tk.Entry(self)
        self.camera_index_entry.pack()

        tk.Button(self, text="Start Camera", command=self.start_camera).pack(pady=5)
        tk.Button(self, text="Stop Camera", command=self.stop_camera).pack(pady=5)

        tk.Label(self, text="Terminal Output:").pack()
        self.terminal_output = scrolledtext.ScrolledText(self, height=15, width=90)
        self.terminal_output.pack(pady=5)

        # Redirect stdout to the terminal output
        sys.stdout = TextRedirector(self.terminal_output, "stdout")
        sys.stderr = TextRedirector(self.terminal_output, "stderr")

    def load_targets(self):
        try:
            self.target_listbox.delete(0, tk.END)
            data = pd.read_csv('Training_data.csv')
            targets = data['target'].unique()
            for target in targets:
                self.target_listbox.insert(tk.END, target)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def process_images(self):
        try:
            num_images = int(self.num_images_entry.get())
            self.processor.process_images(self.processor.trainfolderpath, num_images)
            messagebox.showinfo("Info", "Images processed.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def convert_json_to_csv(self):
        try:
            selected_targets = [self.target_listbox.get(i) for i in self.target_listbox.curselection()]
            self.converter.convert_json_to_csv(selected_targets)
            messagebox.showinfo("Info", "JSON converted to CSV.")
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def prepare_ml_data(self):
        try:
            self.preparer.prepare_ml_data()
            messagebox.showinfo("Info", "ML data prepared.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def train_model(self):
        try:
            data = pd.read_csv('Training_data.csv')
            selected_targets = [self.target_listbox.get(i) for i in self.target_listbox.curselection()]
            if selected_targets:
                data = data[~data['target'].isin(selected_targets)]
            X = data.drop(columns=['target'])
            y = data['target']
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Accuracy: {accuracy * 100:.2f}%")
            print("Classification Report:")
            print(classification_report(y_test, y_pred))
            print("Confusion Matrix:")
            print(confusion_matrix(y_test, y_pred))
            joblib.dump(model, 'random_forest_model.pkl')
            self.model = model
            messagebox.showinfo("Info", "Model trained and saved.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def start_camera(self):
        try:
            if not self.model:
                messagebox.showerror("Error", "Model not found. Please train the model first.")
                return

            camera_index = int(self.camera_index_entry.get())
            self.cap = cv2.VideoCapture(camera_index)
            pTime = 0
            while self.cap.isOpened():
                success, img = self.cap.read()
                if not success:
                    break
                img, faces = self.face_mesh.findFaceMesh(img, draw=True)
                if faces:
                    features = self.preparer.extract_features(faces[0])
                    feature_vector = self.features_to_vector(features)
                    emotion = self.model.predict([feature_vector])[0]
                    cv2.putText(img, f"Emotion: {emotion}", (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

                cTime = time.time()
                fps = 1 / (cTime - pTime)
                pTime = cTime

                cv2.putText(img, f"FPS: {int(fps)}", (10, 20),
                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
                cv2.imshow("Image", img)

                if cv2.waitKey(1) == ord('q'):
                    break

            self.cap.release()
            cv2.destroyAllWindows()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def stop_camera(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
            cv2.destroyAllWindows()

    def features_to_vector(self, features):
        vector = []
        for feature_name, feature_data in features.items():
            for dist_name, dist_value in feature_data.get('distances', {}).items():
                vector.append(dist_value)
            for angle_name, angle_value in feature_data.get('angles', {}).items():
                vector.append(angle_value)
        return vector

class TextRedirector(object):
    def __init__(self, widget, tag):
        self.widget = widget
        self.tag = tag

    def write(self, str):
        self.widget.configure(state="normal")
        self.widget.insert("end", str, (self.tag,))
        self.widget.configure(state="disabled")
        self.widget.see("end")

if __name__ == "__main__":
    app = HandlerUI()
    app.mainloop()