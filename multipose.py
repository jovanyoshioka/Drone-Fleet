# Referenced: https://github.com/nicknochnack/MultiPoseMovenetLightning
# Model: https://tfhub.dev/google/movenet/multipose/lightning/1
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
    for person in keypoints_with_scores:
        draw_connections(frame, person, edges, confidence_threshold)
        draw_keypoints(frame, person, confidence_threshold)

# If using GPU, prevent TensorFlow from consuming all RAM
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Load model
model = hub.load('https://tfhub.dev/google/movenet/multipose/lightning/1')
movenet = model.signatures['serving_default']

# Make detections
cap = cv.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()

    # Resize image
    img = frame.copy()
    # NOTE: Make sure aspect ratio matches frame's aspect ratio, i.e., 192/256 = frameWidth/frameHeight
    # Recommended that larger side is 256 pixels while keeping original aspect ratio.
    # "The size of the input image controls the tradeoff between speed vs. accuracy"
    img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 192, 256)
    input_img = tf.cast(img, dtype=tf.int32)

    # Detection section
    results = movenet(input_img)
    keypoints_with_scores = results['output_0'].numpy()[:, :, :51].reshape((6, 17, 3))

    # Draw bounding boxes; will later compare with color filter to determine commander
    # [ymin, xmin, ymax, xmax, confidence_score], coordinates are normalized.
    bounding_boxes = results['output_0'].numpy()[:, :, 51:56].reshape((6, 5, 1))
    y, x, _ = frame.shape
    for bb in bounding_boxes:
        if bb[4][0] > 0.3:
            pt1 = (int(x * bb[1][0]), int(y * bb[0][0]))
            pt2 = (int(x * bb[3][0]), int(y * bb[2][0]))
            cv.rectangle(frame, pt1, pt2, (255, 0, 0), 2)

    # Render keypoints
    loop_through_people(frame, keypoints_with_scores, EDGES, 0.3) # 0.3 is confidence threshold

    cv.imshow('Multipose', frame)

    if cv.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()