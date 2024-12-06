import cv2
import mediapipe as mp
import time

# mpDraw = mp.solutions.drawing_utils
# mpFaceMesh = mp.solutions.face_mesh
# faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
# drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2)


# while True:
#     success, img = cap.read()


#     imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     results = faceMesh.process(imgRGB)
#     if results.multi_face_landmarks:
#         for faceLms in results.multi_face_landmarks:
#             mpDraw.draw_landmarks(
#                 img, faceLms, mpFaceMesh.FACEMESH_TESSELATION,
#                 mpDraw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
#                 mpDraw.DrawingSpec(color=(0, 0, 255), thickness=1)
#             )

#             for id ,lm in enumerate(faceLms.landmark):
#                 ih, iw, ic = img.shape
#                 x, y = int(lm.x*iw), int(lm.y*ih)
#                 #print(id, x, y)



class faceMeshModule:
    def __init__(self, staticMode=False, maxFaces=2, minDetectionCon=0.5, minTrackCon=0.5):
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(static_image_mode=staticMode, max_num_faces=maxFaces, min_detection_confidence=minDetectionCon, min_tracking_confidence=minTrackCon)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)

    def findFaceMesh(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.faceMesh.process(imgRGB)
        faces = []
        if results.multi_face_landmarks:
            for faceLms in results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        img, faceLms, self.mpFaceMesh.FACEMESH_TESSELATION,
                        self.mpDraw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=0),
                        self.mpDraw.DrawingSpec(color=(0, 255, 0), thickness=1)
                    )
                face = []
                for id ,lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x*iw), int(lm.y*ih)
                    cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN, 0.3, (255, 0,255), 1)
                    face.append([x, y])
                faces.append(face)
        return img, faces if faces else None







def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    tracking = faceMeshModule()

    while True:
        success, img = cap.read()
        img, faces = tracking.findFaceMesh(img, True)
        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime
        try:
            if len(faces) != 0:
                print(len(faces))
        except:
            print("No face detected")
        cv2.putText(img, "FPS: " + str(int(fps)), (0, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()


