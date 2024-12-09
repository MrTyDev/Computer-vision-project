import os
import json
import matplotlib.pyplot as plt

def select_and_display_face_data():
    # Define emotions and base path
    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    base_path = 'processed_data'
    
    # Display emotion options
    print("\nSelect emotion folder:")
    for i, emotion in enumerate(emotions, 1):
        print(f"{i}. {emotion}")
    
    # Get emotion choice
    while True:
        try:
            emotion_choice = int(input("\nEnter number (1-7): "))
            if 1 <= emotion_choice <= 7:
                break
            print("Please enter a number between 1 and 7")
        except ValueError:
            print("Please enter a valid number")
    
    # Get selected emotion folder path
    selected_emotion = emotions[emotion_choice-1]
    emotion_path = os.path.join(base_path, selected_emotion)
    
    # List available face data files
    face_files = [f for f in os.listdir(emotion_path) if f.startswith('face') and f.endswith('data.json')]
    face_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))  # Sort numerically
    
    print(f"\nAvailable face data files in {selected_emotion}:")
    for i, file in enumerate(face_files, 1):
        print(f"{i}. {file}")
    
    # Get file choice
    while True:
        try:
            file_choice = int(input(f"\nEnter number (1-{len(face_files)}): "))
            if 1 <= file_choice <= len(face_files):
                break
            print(f"Please enter a number between 1 and {len(face_files)}")
        except ValueError:
            print("Please enter a valid number")
    
    # Load and display selected file
    selected_file = face_files[file_choice-1]
    file_path = os.path.join(emotion_path, selected_file)
    
    with open(file_path, 'r') as f:
        face_data = json.load(f)
    
    # Extract coordinates and plot
    data = face_data[0]  # First face in the data
    x = [point[1] for point in data]
    y = [point[2] for point in data]
    
    plt.figure(figsize=(8, 8))
    plt.scatter(x, y)
    plt.title(f"{selected_emotion} - {selected_file}")
    plt.gca().invert_yaxis()  # Invert Y axis to match face orientation
    plt.show()

if __name__ == "__main__":
    select_and_display_face_data()