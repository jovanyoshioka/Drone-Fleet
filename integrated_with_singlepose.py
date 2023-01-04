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

POSE_CONFIDENCE_THRESH = 0.2
VERTICAL_ALIGN_THRESH = 0.05 # multiplied by frame width


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


# Determines if top and bottom points are vertically aligned, verifying that top point is above bottom point as well
# pt format: [y, x, score]
def is_vertically_aligned(top_pt, bottom_pt, frame_width):
    # Note: above point has lesser y-value since top left of frame is (0, 0)
    return top_pt[0] < bottom_pt[0] and abs(top_pt[1] - bottom_pt[1]) < frame_width * VERTICAL_ALIGN_THRESH


# Detect commander's gestures
# Gestures supported: hands up/down
# keypoints_with_scores: [nose, left eye, right eye, left ear, right ear, left shoulder, right shoulder, left elbow,
#   right elbow, left wrist, right wrist, left hip, right hip, left knee, right knee, left ankle, right ankle].
def detect_gestures(keypoints_with_scores, frame):
    frame_height, frame_width, _ = frame.shape
    keypoints_with_scores = np.multiply(keypoints_with_scores, [frame_height, frame_width, 1])

    left_wrist = keypoints_with_scores[9]
    left_elbow = keypoints_with_scores[7]

    right_wrist = keypoints_with_scores[10]
    right_elbow = keypoints_with_scores[8]

    # Left hand up or down respectively, verify left wrist/elbow confidence score.
    if left_wrist[2] > POSE_CONFIDENCE_THRESH and left_elbow[2] > POSE_CONFIDENCE_THRESH:
        if is_vertically_aligned(left_wrist, left_elbow, frame_width):
            cv.putText(frame, "LEFT HAND UP", (10, 25), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif is_vertically_aligned(left_elbow, left_wrist, frame_width):
            cv.putText(frame, "LEFT HAND DOWN", (10, 25), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Right hand up or down respectively, verify right wrist/elbow confidence score.
    if right_wrist[2] > POSE_CONFIDENCE_THRESH and right_elbow[2] > POSE_CONFIDENCE_THRESH:
        if is_vertically_aligned(right_wrist, right_elbow, frame_width):
            cv.putText(frame, "RIGHT HAND UP", (10, 55), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif is_vertically_aligned(right_elbow, right_wrist, frame_width):
            cv.putText(frame, "RIGHT HAND DOWN", (10, 55), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


def main():
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
        draw_connections(frame, keypoints_with_scores, EDGES, POSE_CONFIDENCE_THRESH)
        draw_keypoints(frame, keypoints_with_scores, POSE_CONFIDENCE_THRESH)

        detect_gestures(keypoints_with_scores, frame)

        cv.imshow('SinglePose', frame)

        if cv.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

main()
