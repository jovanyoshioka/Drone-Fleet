# Referenced: https://github.com/nicknochnack/MultiPoseMovenetLightning
# Model: https://tfhub.dev/google/movenet/multipose/lightning/1
# Notes:
#   - Best for "detecting fitness/fast movement with motion blur poses"
#   - Best when 3ft - 6ft away from camera

import tensorflow as tf
import tensorflow_hub as hub
import cv2 as cv
import numpy as np


class Commander:
    def __init__(self, overlap_area, overlap_pts, bb_pts, keypoints_with_scores):
        self.overlap_area = overlap_area
        self.overlap_pts = overlap_pts
        self.bb_pts = bb_pts
        self.keypoints_with_scores = keypoints_with_scores


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


# arg format: [ymin, xmin, ymax, xmax], not normalized
# return: area, overlap_pt1, overlap_pt2
def calc_overlapping_area(bb1, bb2):
    width = min(bb1[3], bb2[3]) - max(bb1[1], bb2[1])
    height = min(bb1[2], bb2[2]) - max(bb1[0], bb2[0])

    # Draw overlapping area
    pt1 = [max(bb1[1], bb2[1]), max(bb1[0], bb2[0])]
    pt2 = [min(bb1[3], bb2[3]), min(bb1[2], bb2[2])]

    if width > 0 and height > 0:
        return width * height, pt1, pt2
    else:
        return 0, [0, 0], [0, 0]



def filter_color(frame):
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Red HSV
    l_b = np.array([150, 85, 80])
    u_b = np.array([180, 255, 255])

    mask = cv.inRange(hsv, l_b, u_b)

    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE, (10, 10)))
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (20, 20)))

    # Object detection
    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=lambda x: cv.contourArea(x))
        x, y, w, h = cv.boundingRect(cnt)
        return [y, x, y + h, x + w]
    else:
        return [0, 0, 0, 0]


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
    model = hub.load('https://tfhub.dev/google/movenet/multipose/lightning/1')
    movenet = model.signatures['serving_default']

    # Make detections
    cap = cv.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()

        # Filter commander's color and get bounding box,
        # [ymin, xmin, ymax, xmax]
        color_bb_pts = filter_color(frame)

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

        # Initialize commander instance to hold necessary information for tracking
        # area, overlap [ymin, xmin, ymax, xmax], bb [ymin, xmin, ymax, xmax]
        commander = Commander(-1, [0, 0, 0, 0], [0, 0, 0, 0], [])

        # Draw bounding boxes; will later compare with color filter to determine commander
        # [ymin, xmin, ymax, xmax, confidence_score], coordinates are normalized.
        bounding_boxes = results['output_0'].numpy()[:, :, 51:56].reshape((6, 5, 1))
        y, x, _ = frame.shape
        for i, bb in enumerate(bounding_boxes):
            if bb[4][0] > POSE_CONFIDENCE_THRESH:
                bb_pts = [int(y * bb[0][0]), int(x * bb[1][0]), int(y * bb[2][0]), int(x * bb[3][0])]

                if color_bb_pts[0] != 0 or color_bb_pts[1] != 0 or color_bb_pts[2] != 0 or color_bb_pts[3] != 0:
                    area, overlap_pt1, overlap_pt2 = calc_overlapping_area(bb_pts, color_bb_pts)
                    if area > commander.overlap_area and area > (
                            (bb_pts[3] - bb_pts[1]) * (bb_pts[2] - bb_pts[0])) * 0.2:
                        # Draw rectangle of previous, if not default
                        if commander.overlap_area != -1:
                            cv.rectangle(frame, (commander.bb_pts[1], commander.bb_pts[0]),
                                         (commander.bb_pts[3], commander.bb_pts[2]), (255, 0, 0), 2)

                        commander.overlap_area = area
                        commander.overlap_pts = [overlap_pt1[1], overlap_pt1[0], overlap_pt2[1], overlap_pt2[0]]
                        commander.bb_pts = bb_pts
                        commander.keypoints_with_scores = keypoints_with_scores[i]
                    else:
                        cv.rectangle(frame, (bb_pts[1], bb_pts[0]), (bb_pts[3], bb_pts[2]), (255, 0, 0), 2)
                else:
                    cv.rectangle(frame, (bb_pts[1], bb_pts[0]), (bb_pts[3], bb_pts[2]), (255, 0, 0), 2)

        if commander.overlap_area != -1:
            cv.rectangle(frame, (commander.bb_pts[1], commander.bb_pts[0]), (commander.bb_pts[3], commander.bb_pts[2]),
                         (0, 255, 0), 2)
            # Create overlap overlay
            overlay = frame.copy()
            cv.rectangle(frame, (commander.overlap_pts[1], commander.overlap_pts[0]),
                         (commander.overlap_pts[3], commander.overlap_pts[2]), (255, 0, 150), -1)
            alpha = 0.2
            frame = cv.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # Render keypoints
        loop_through_people(frame, keypoints_with_scores, EDGES, POSE_CONFIDENCE_THRESH)

        # If commander found, detect body gestures
        if commander.overlap_area != -1:
            detect_gestures(commander.keypoints_with_scores, frame)

        cv.imshow('Multipose', frame)

        if cv.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

main()