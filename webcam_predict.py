import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from eye_crop_test import detect_best_eye

# ================================
# Load Trained Model
# ================================
MODEL_PATH = "best_model.h5"
model = load_model(MODEL_PATH)

class_names = ["Closed", "Open", "no_yawn", "yawn"]

print("🎉 Model Loaded Successfully! Starting Webcam...")

# ================================
# Predict on Single Frame
# ================================
def predict_frame(frame):

    if frame is None:
        return None, None, "Frame Error"

    # Convert BGR → RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (224, 224))

    input_img = np.expand_dims(resized, axis=0)
    input_img = preprocess_input(input_img)

    # Face Prediction
    pred_face = model.predict(input_img, verbose=0)
    face_class = class_names[np.argmax(pred_face)]
    face_conf = float(np.max(pred_face))

    # ================================
    # Eye Detection
    # ================================
    cv2.imwrite("temp_frame.jpg", frame)
    best_eye, _, _, _ = detect_best_eye("temp_frame.jpg")

    if best_eye is not None:
        eye_rgb = cv2.cvtColor(best_eye, cv2.COLOR_BGR2RGB)
        eye_resized = cv2.resize(eye_rgb, (224, 224))
        eye_input = np.expand_dims(eye_resized, axis=0)
        eye_input = preprocess_input(eye_input)

        pred_eye = model.predict(eye_input, verbose=0)
        eye_class = class_names[np.argmax(pred_eye)]
        eye_conf = float(np.max(pred_eye))
    else:
        # No eye detected fallback
        eye_class = face_class
        eye_conf = face_conf

    # ================================
    # Drowsiness Logic
    # ================================
    if eye_class == "Closed" or face_class == "yawn":
        state = "😴 Drowsy"
    else:
        state = "😎 Non-Drowsy"

    return face_class, eye_class, state


# ================================
# WEBCAM LIVE FEED
# ================================
cap = cv2.VideoCapture(0)  # 0 = Laptop webcam

if not cap.isOpened():
    print("❌ Could not access webcam!")
    exit()

print("📸 Webcam Started — Press 'Q' to Quit")

while True:
    ret, frame = cap.read()

    if not ret:
        print("⚠ Frame not captured!")
        continue

    face_class, eye_class, state = predict_frame(frame)

    # Draw results on frame
    cv2.putText(frame, f"Face: {face_class}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.putText(frame, f"Eye: {eye_class}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.putText(frame, f"State: {state}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("Driver Drowsiness Detection - LIVE", frame)

    # Exit on Q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("🛑 Webcam Closed.")