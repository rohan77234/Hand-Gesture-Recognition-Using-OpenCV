import cv2
import mediapipe as mp
import time

# """This is a Python script that uses OpenCV and MediaPipe libraries to detect and track hand gestures from a live video stream. The script creates a handDetector class that initializes the MediaPipe hands object and provides methods to find hands and their positions in each frame. The findHands method detects the landmarks on the hand and draws them on the image using the mpDraw utility. The findPosition method returns a list of landmark positions for a specific hand, which can be used to identify hand gestures.

# The main function reads frames from the camera, calls the findHands and findPosition methods to detect the landmarks and positions, and displays the image with the detected landmarks. The script also calculates and displays the frames per second (FPS) rate for the video stream.

# To use this script, you need to install OpenCV and MediaPipe libraries and run the script. You can also modify the script to recognize and perform actions based on specific hand gestures by mapping landmark positions to gestures."""
class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True, color =  (255, 0, 255), z_axis=False):

        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
             #   print(id, lm)
                h, w, c = img.shape
                if z_axis == False:
                   cx, cy = int(lm.x * w), int(lm.y * h)
                    # print(id, cx, cy)
                   lmList.append([id, cx, cy])
                elif z_axis:
                    cx, cy, cz = int(lm.x * w), int(lm.y * h), round(lm.z,3)
                    # print(id, cx, cy, cz)
                    lmList.append([id, cx, cy, cz])

                if draw:
                    cv2.circle(img, (cx, cy),5,color, cv2.FILLED)

        return lmList


def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(1)
    detector = handDetector(maxHands=1)
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img,z_axis=True,draw=False)
        if len(lmList) != 0:
            print(lmList[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()



    # Thank you for the clarification. The code snippet you provided earlier is a part of the HandTrackingModule.py module. This module defines the handDetector class, which provides methods to detect and track hands in a video stream using the MediaPipe library.

# The __init__ method initializes the hands and mpDraw objects and sets the default parameters for detection and tracking. The findHands method processes each frame of the video stream, detects and draws the hand landmarks using mpDraw. The findPosition method returns the landmarks' positions for a specified hand in the image, which can be used to identify hand gestures.

# The main function is not a part of this module and is likely present in the main script that imports this module. It uses the handDetector class to process the video stream and detect the landmarks' positions.

# Overall, this module provides a reusable code structure to detect and track hand gestures using the MediaPipe library.