import cv2
import mediapipe as mp
import time
import numpy as np

class faceMeshModule:
    def __init__(self, staticMode=False, maxFaces=2, minDetectionCon=0.5, minTrackCon=0.5):
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(
            static_image_mode=staticMode,
            max_num_faces=maxFaces,
            min_detection_confidence=minDetectionCon,
            min_tracking_confidence=minTrackCon
        )
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)
        self.MOUTH_LANDMARKS = [0, 267, 269, 270, 409, 306, 375, 321,
                                405, 314, 17, 84, 181, 91, 146, 61,
                                185, 40, 39, 37]
        self.FACE_LANDMARKS = [10, 338, 297, 332, 284, 251, 389, 356,
                               454, 323, 361, 288, 397, 365, 379, 378,
                               400, 377, 152, 148, 176, 149, 150, 136,
                               172, 58, 132, 93, 234, 127, 162, 21,
                               54, 103, 67, 109]
        self.RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155,
                          133, 246, 161, 160, 159, 158, 157, 173]
        self.LEFT_EYE = [362, 382, 398, 384, 385, 386, 387, 263,
                         249, 390, 373, 374, 380, 381]
        self.RIGHT_EYEBROW = [46, 53, 52, 65, 55]
        self.LEFT_EYEBROW = [285, 293, 300, 296, 334]
        self.NOSE = [168, 6, 197, 195, 5, 1, 2, 3, 4, 19, 44, 274, 440, 220, 344, 278, 294, 115, 48, 64]
        self.RIGHT_CHEEK = [50, 117, 118, 101, 36, 123, 147, 213, 192, 214, 202]
        self.LEFT_CHEEK = [349, 348, 347, 330, 280, 411, 416, 434, 422, 424]
        self.ALL_LANDMARKS = (self.MOUTH_LANDMARKS + self.FACE_LANDMARKS + self.RIGHT_EYE +
                              self.LEFT_EYE + self.RIGHT_EYEBROW + self.LEFT_EYEBROW +
                              self.NOSE + self.RIGHT_CHEEK + self.LEFT_CHEEK)

    def findFaceMesh(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.faceMesh.process(imgRGB)
        faces = []
        if results.multi_face_landmarks:
            ih, iw, ic = img.shape
            for faceLms in results.multi_face_landmarks:
                # Collect landmarks
                x_list = []
                y_list = []
                for idx in self.ALL_LANDMARKS:
                    x = faceLms.landmark[idx].x * iw
                    y = faceLms.landmark[idx].y * ih
                    x_list.append(x)
                    y_list.append(y)

                # Compute bounding box
                x_min, x_max = min(x_list), max(x_list)
                y_min, y_max = min(y_list), max(y_list)

                # Normalize landmarks to 48x48 area
                scale_x = 48 / (x_max - x_min)
                scale_y = 48 / (y_max - y_min)
                scale = min(scale_x, scale_y)  # Keep aspect ratio

                face_normalized = []
                for x, y in zip(x_list, y_list):
                    x_norm = (x - x_min) * scale
                    y_norm = (y - y_min) * scale
                    face_normalized.append((int(x_norm), int(y_norm)))

                # Optionally draw normalized landmarks on a separate image
                normalized_img = np.zeros((48, 48, 3), dtype=np.uint8)
                for point in face_normalized:
                    cv2.circle(normalized_img, point, 1, (255, 255, 255), -1)

                # Draw landmarks on the original image if needed
                if draw:
                    for x, y in zip(x_list, y_list):
                        cv2.circle(img, (int(x), int(y)), 2, (0, 0, 255), -1)

                faces.append(face_normalized)

                # Display the normalized face
                cv2.imshow("Normalized Face", normalized_img)

        return img, faces if faces else None

def main(input_source=None):
    pTime = 0
    tracking = faceMeshModule()

    if input_source:
        # Process image file
        img = cv2.imread(input_source)
        img, faces = tracking.findFaceMesh(img, draw=True)

        # Display the image
        cv2.imshow("Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Use 'faces' data for model training
        if faces:
            # Save faces data or process further
            pass
    else:
        # Process webcam feed
        cap = cv2.VideoCapture(0)
        while True:
            success, img = cap.read()
            img, faces = tracking.findFaceMesh(img, draw=True)
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime

            cv2.putText(img, f"FPS: {int(fps)}", (10, 20),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
            cv2.imshow("Image", img)
            if cv2.waitKey(1) == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # For image input, provide the image path
    main("C:/Users/tyhug/Desktop/Computer_vision_MLProject/Computer-vision-project/train/disgust/Training_680349.jpg")
    

    # For webcam input, call main without arguments
    #main()