import cv2
import numpy as np
import mediapipe as mp
import time
import zmq
from tensorflow.keras.models import load_model

# =========================================================
# MEDIAPIPE SETUP
# =========================================================
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def draw_styled_landmarks(image, results):

    mp_drawing.draw_landmarks(
        image,
        results.left_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
    )

    mp_drawing.draw_landmarks(
        image,
        results.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=2)
    )


def extract_keypoints(results):
    lh = np.array([[p.x, p.y, p.z] for p in results.left_hand_landmarks.landmark]).flatten() \
        if results.left_hand_landmarks else np.zeros(21 * 3)

    rh = np.array([[p.x, p.y, p.z] for p in results.right_hand_landmarks.landmark]).flatten() \
        if results.right_hand_landmarks else np.zeros(21 * 3)

    return np.concatenate([lh, rh])


# =========================================================
# MODEL
# =========================================================
model = load_model("training/best_model_bigru.h5")

actions = np.array([
    'up','down','left','right','go','back',
    'takeoff','stop','one','two','three','four',
    'five','six','seven','eight','nine',
    'A','B','C','H','I','L','O','R','U'
])

# =========================================================
# ZMQ
# =========================================================
context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")

# =========================================================
# PARAMETERS
# =========================================================
SEQ_LEN = 20
PRED_INTERVAL = 25.0      # seconds
THRESHOLD = 0.6

sequence = []
sentence = []

gesture_timer_start = time.time()

cap = cv2.VideoCapture(0)

# =========================================================
# MAIN LOOP
# =========================================================
with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as holistic:

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        image, results = mediapipe_detection(frame, holistic)
        draw_styled_landmarks(image, results)

        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-SEQ_LEN:]

        current_time = time.time()
        elapsed = current_time - gesture_timer_start
        remaining = max(0, PRED_INTERVAL - elapsed)

        # =====================================================
        # PREDICTION EVERY 25 SECONDS
        # =====================================================
        if len(sequence) == SEQ_LEN and elapsed >= PRED_INTERVAL:

            res = model.predict(np.expand_dims(sequence, axis=0))[0]

            idx = np.argmax(res)
            confidence = res[idx]
            gesture = actions[idx]

            print(f"Predicted: {gesture} | confidence: {confidence:.3f}")

            if confidence >= THRESHOLD:

                sentence.append(gesture)
                sentence = sentence[-5:]

                print("SENT:", gesture)

                try:
                    socket.send_string(gesture)
                    reply = socket.recv_string()
                    print("Controller:", reply)

                except Exception as e:
                    print("ZMQ error:", e)

            else:
                print("rejected (low confidence)")

            # RESET TIMER AFTER EACH CYCLE
            gesture_timer_start = current_time

        # =====================================================
        # UI
        # =====================================================
        cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)

        cv2.putText(
            image,
            ' '.join(sentence),
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            image,
            f"Next gesture in: {int(remaining)}s",
            (350, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.imshow("Gesture Recognition", image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
