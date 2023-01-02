import cv2
import numpy as np

# For tuning color filtering
hsvBar = False


# Dummy function for getTrackbarPos
def nothing():
	return


def process_frame(frame):
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	if hsvBar:
		l_h = cv2.getTrackbarPos("LH", "Tracking")
		l_s = cv2.getTrackbarPos("LS", "Tracking")
		l_v = cv2.getTrackbarPos("LV", "Tracking")

		u_h = cv2.getTrackbarPos("UH", "Tracking")
		u_s = cv2.getTrackbarPos("US", "Tracking")
		u_v = cv2.getTrackbarPos("UV", "Tracking")

	# Red HSV
	if not hsvBar:
		l_b = np.array([150, 85, 80])
		u_b = np.array([180, 255, 255])
	else:
		l_b = np.array([l_h, l_s, l_v])
		u_b = np.array([u_h, u_s, u_v])

	mask = cv2.inRange(hsv, l_b, u_b)

	# cv2.imshow("mask", mask)
	mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10)))
	# cv2.imshow("less noise", mask)
	mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20)))
	# cv2.imshow("filled gaps", mask)

	# Object detection
	contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	if contours:
		cnt = max(contours, key=lambda x: cv2.contourArea(x))
		x, y, w, h = cv2.boundingRect(cnt)
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

	res = cv2.bitwise_and(frame, frame, mask=mask)

	cv2.imshow("frame", frame)
	cv2.imshow("mask", mask)
	cv2.imshow("res", res)


def main():
	cap = cv2.VideoCapture(0)

	if hsvBar:
		cv2.namedWindow("Tracking")
		cv2.createTrackbar("LH", "Tracking", 0, 255, nothing)
		cv2.createTrackbar("LS", "Tracking", 0, 255, nothing)
		cv2.createTrackbar("LV", "Tracking", 0, 255, nothing)
		cv2.createTrackbar("UH", "Tracking", 255, 255, nothing)
		cv2.createTrackbar("US", "Tracking", 255, 255, nothing)
		cv2.createTrackbar("UV", "Tracking", 255, 255, nothing)

	while True:
		_, frame = cap.read()

		process_frame(frame)

		key = cv2.waitKey(1)
		if key == 27:
			break

	cap.release()
	cv2.destroyAllWindows()

main()
