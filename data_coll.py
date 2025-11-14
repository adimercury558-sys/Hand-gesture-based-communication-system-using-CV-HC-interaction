#Deeraj_47_github @HAKAISHIN_KAGGLE
import cv2
import mediapipe as mp
import numpy as np
import csv
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

GESTURES = ['hello', 'how', 'eat', 'sleep', 'write', 'yes', 'no', 'thankyou', 'good', 'morning']

if not os.path.exists("sign_data.csv"):
    with open("sign_data.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label"] + [f"{i}" for i in range(63)])  # 21 landmarks * 3 coords

cap = cv2.VideoCapture(0)

for gesture in GESTURES:
    print(f"Show gesture '{gesture}' and press 's' to save samples.")
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        cv2.putText(frame, f"Gesture: {gesture}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("Collect Data", frame)

        key = cv2.waitKey(1)
        if key == ord('s') and result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                data = []
                for lm in hand_landmarks.landmark:
                    data.extend([lm.x, lm.y, lm.z])
                with open("sign_data.csv", "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([gesture] + data)
            print(f"Saved {gesture}")
        elif key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
