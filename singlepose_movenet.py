# Referenced: https://github.com/nicknochnack/MultiPoseMovenetLightning
# Model: https://tfhub.dev/google/movenet/singlepose/thunder/4
# Notes:
#   - Best for "detecting fitness/fast movement with motion blur poses"
#   - Best when 3ft - 6ft away from camera

import tensorflow as tf
import tensorflow_hub as hub
import cv2 as cv
import numpy as np

EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}


# Draw edges
def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]

        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            cv.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)


# Draw keypoints
def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv.circle(frame, (int(kx), int(ky)), 6, (0, 255, 0), -1)


# Function to loop through each person detected and render
def loop_through_people(frame, keypoints_with_scores, edges, confidence_threshold):
    # for person in keypoints_with_scores:
    draw_connections(frame, keypoints_with_scores, edges, confidence_threshold)
    draw_keypoints(frame, keypoints_with_scores, confidence_threshold)


# If using GPU, prevent TensorFlow from consuming all RAM
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Load model
model = hub.load('https://tfhub.dev/google/movenet/singlepose/thunder/4')
movenet = model.signatures['serving_default']

# Make detections
cap = cv.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()

    # Since tf image must be 256x256, resize to square dimensions.
    frame = cv.resize(frame, (512, 512))

    # Resize image
    img = frame.copy()
    # Must be 256x256
    img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 256, 256)
    input_img = tf.cast(img, dtype=tf.int32)

    # Detection section
    results = movenet(input_img)
    keypoints_with_scores = results['output_0'].numpy()[:, :, :, :51].reshape((17, 3))

    # Render keypoints
    loop_through_people(frame, keypoints_with_scores, EDGES, 0.2) # 0.3 is default confidence threshold

    cv.imshow('Singlepose', frame)

    if cv.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
