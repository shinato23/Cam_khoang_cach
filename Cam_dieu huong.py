import cv2
import numpy as np
import serial
import time

def is_vertical_obstacle(region, min_height=15):
    height, width = region.shape
    vertical_lines = 0
    for col in range(width):
        col_data = region[:, col]
        white_pixels = np.where(col_data == 255)[0]
        if len(white_pixels) == 0:
            continue
        continuous = np.split(white_pixels, np.where(np.diff(white_pixels) != 1)[0]+1)
        max_len = max([len(group) for group in continuous])
        if max_len >= min_height:
            vertical_lines += 1
    return vertical_lines

def detect_direction_and_speed(frame):
    frame = cv2.resize(frame, (320, 240))
    height, width = frame.shape[:2]

    roi = frame[int(height * 0.8):, :]  # Lấy vùng ROI gần robot
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY_INV)

    third = width // 3
    left = thresh[:, 0:third]
    center = thresh[:, third:2*third]
    right = thresh[:, 2*third:width]

    left_score = is_vertical_obstacle(left)
    center_score = is_vertical_obstacle(center)
    right_score = is_vertical_obstacle(right)

    print(f"Obstacle lines -> Left: {left_score}, Center: {center_score}, Right: {right_score}")

    close_threshold = 15
    margin = 3

    direction = "STRAIGHT"
    steering_angle = 0
    speed = 70

    if center_score > close_threshold:
        speed = 30
        if right_score > left_score + margin:
            direction = "RIGHT"
            steering_angle = +30
        elif left_score > right_score + margin:
            direction = "LEFT"
            steering_angle = -30
        else:
            direction = "BACK"
            steering_angle = 0
    return direction, steering_angle, speed, thresh

def send_command_to_serial(direction):
    if direction == "LEFT":
        ser.write(b'L')
    elif direction == "RIGHT":
        ser.write(b'R')
    elif direction == "STRAIGHT":
        ser.write(b'F')
    elif direction == "BACK":
        ser.write(b'B')

# Thiết lập Serial
ser = serial.Serial('COM3', 9600, timeout=1)
time.sleep(2)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    direction, angle, speed, thresh = detect_direction_and_speed(frame)
    send_command_to_serial(direction)

    # Hiển thị kết quả
    cv2.putText(frame, f"Direction: {direction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(frame, f"Angle: {angle}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
    cv2.putText(frame, f"Speed: {speed}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    third = frame.shape[1] // 3
    cv2.line(frame, (third, 0), (third, frame.shape[0]), (255, 0, 0), 1)
    cv2.line(frame, (2*third, 0), (2*third, frame.shape[0]), (255, 0, 0), 1)

    cv2.imshow("Camera View", frame)
    cv2.imshow("Threshold", thresh)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
ser.close()
