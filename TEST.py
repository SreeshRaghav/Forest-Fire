import cv2
import numpy as np
from PIL import Image
import tensorflow as tf

# Load the pre-trained fire detection model
model = tf.keras.models.load_model('my_model.h5')

# Initialize video capture from webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame from the video stream
    ret, frame = cap.read()

    if not ret:
        print("Error: Unable to capture frame from video stream")
        break

    # Convert the captured frame to RGB format (assuming model expects RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR to RGB conversion

    # Resize the frame to the size the model was trained on
    resized_frame = cv2.resize(rgb_frame, (224, 224))

    # Preprocess the frame (convert to NumPy array and normalize)
    image_array = np.expand_dims(np.array(resized_frame) / 255.0, axis=0)

    # Get predictions from the fire detection model
    predictions = model.predict(image_array)[0]
    predicted_class = np.argmax(predictions)

    # Implement fire detection logic based on model prediction
    if predicted_class == 0:  # Assuming class 0 indicates fire
        # Display frame in grayscale (or perform other fire indication actions)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Fire Detection", gray_frame)

        # Print additional information about the prediction (optional)
        print(f"Predicted Class: Fire (Confidence: {predictions[predicted_class]:.2f})")
    else:
        # Display frame in color (or perform other non-fire actions)
        cv2.imshow("Fire Detection", frame)

    # Exit on 'q' press
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
