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

def process_frame_for_obstacles(frame):
    frame = cv2.resize(frame, (320, 240))
    height, width = frame.shape[:2]
    roi = frame[int(height * 0.8):, :]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY_INV)
    return thresh

def detect_direction_and_speed(frame, hold_direction=None):
    frame = cv2.resize(frame, (320, 240))
    height, width = frame.shape[:2]
    roi = frame[int(height * 0.8):, :]
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

    if hold_direction is not None:
        direction = hold_direction
        speed = 30
    elif center_score > close_threshold:
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
        ser.write(bytes([0x02]))
    elif direction == "RIGHT":
        ser.write(bytes([0x03]))
    elif direction == "STRAIGHT":
        ser.write(bytes([0x01]))
    elif direction == "BACK":
        ser.write(bytes([0x04]))

# Thiết lập Serial
ser = serial.Serial('/dev/ttyUSB0', 152000, timeout=1)
time.sleep(2)

cap = cv2.VideoCapture(0)
time.sleep(2)

# 🚗 Quét ban đầu trái phải
print("Scanning left...")
send_command_to_serial("LEFT")
time.sleep(1)
ret, frame = cap.read()
left_thresh = process_frame_for_obstacles(frame)
left_score = is_vertical_obstacle(left_thresh[:, :left_thresh.shape[1]//2])

print("Scanning right...")
send_command_to_serial("RIGHT")
time.sleep(1)
ret, frame = cap.read()
right_thresh = process_frame_for_obstacles(frame)
right_score = is_vertical_obstacle(right_thresh[:, right_thresh.shape[1]//2:])

# 🔁 Trở lại trạng thái thẳng để bắt đầu
send_command_to_serial("STRAIGHT")
time.sleep(0.5)

# 🧠 Chọn hướng ưu tiên ban đầu
preferred_direction = "STRAIGHT"
if left_score < right_score:
    preferred_direction = "LEFT"
elif right_score < left_score:
    preferred_direction = "RIGHT"
print(f"Preferred start direction: {preferred_direction}")

# ✅ Vòng lặp chính
hold_start_time = None
hold_duration = 1.0
current_hold = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Kiểm tra giữ hướng
    if current_hold and (time.time() - hold_start_time < hold_duration):
        direction, angle, speed, thresh = detect_direction_and_speed(frame, hold_direction=current_hold)
    else:
        direction, angle, speed, thresh = detect_direction_and_speed(frame)
        if direction in ["LEFT", "RIGHT"]:
            current_hold = direction
            hold_start_time = time.time()
        else:
            current_hold = None

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
