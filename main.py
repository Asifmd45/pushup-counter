import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)

count = 0
stage = None
feedback = ""

# Simple angle calculation using vectors
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (
        np.linalg.norm(ba) * np.linalg.norm(bc)
    )
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB for MediaPipe
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image)

    # Convert back to BGR
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Left arm landmarks (normalized coordinates)
        shoulder = [landmarks[11].x, landmarks[11].y]
        elbow = [landmarks[13].x, landmarks[13].y]
        wrist = [landmarks[15].x, landmarks[15].y]

        angle = calculate_angle(shoulder, elbow, wrist)

        # ---- PUSH-UP LOGIC ----
        if angle < 90:
            stage = "down"

        if angle > 160 and stage == "down":
            stage = "up"
            count += 1
            feedback = "Good Push-Up"

        # ---- FORM CHECK ----
        if angle > 100 and stage == "down":
            feedback = "Go Lower"

        if angle < 160 and stage == "up":
            feedback = "Extend Arms Fully"

        # Display angle
        cv2.putText(image, f"Angle: {int(angle)}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)

        # Draw pose
        mp_draw.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

    # UI Panel
    cv2.rectangle(image, (0, 0), (300, 120), (0, 0, 0), -1)

    cv2.putText(image, f"Push-Ups: {count}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    cv2.putText(image, f"Feedback: {feedback}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 255), 2)

    cv2.imshow("Beginner Push-Up Counter", image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
