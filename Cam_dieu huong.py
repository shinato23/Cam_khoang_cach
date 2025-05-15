import cv2
import numpy as np

def detect_direction_and_speed(frame):
    frame = cv2.resize(frame, (320, 240))
    height, width = frame.shape[:2]

    roi = frame[int(height * 0.6):, :]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)

    third = width // 3
    left = thresh[:, :third]
    center = thresh[:, third:2*third]
    right = thresh[:, 2*third:]

    left_score = np.sum(left == 255)
    center_score = np.sum(center == 255)
    right_score = np.sum(right == 255)

    # Ngưỡng để coi là có vật cản
    close_threshold = 1000

    print(f"Scores -> Left: {left_score}, Center: {center_score}, Right: {right_score}")

    # Mặc định
    direction = "STRAIGHT"
    steering_angle = 0
    speed = 70

    # Trường hợp đặc biệt: vật cản ở cả 2 bên
    if left_score > close_threshold and right_score > close_threshold:
        direction = "REVERSE"
        speed = -30
        steering_angle = 0  # hoặc nhẹ trái/phải nếu cần
        return direction, steering_angle, speed

    # Nếu có vật cản phía trước quá gần
    if center_score > close_threshold:
        if left_score > right_score:
            direction = "RIGHT"
            steering_angle = +30
        elif right_score > left_score:
            direction = "LEFT"
            steering_angle = -30
        else:
            direction = "STRAIGHT"
            steering_angle = 0
        speed = 30
    else:
        direction = "STRAIGHT"
        steering_angle = 0
        speed = 70

    return direction, steering_angle, speed

# Thử nghiệm với webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    direction, angle, speed = detect_direction_and_speed(frame)

    # Hiển thị kết quả lên hình
    cv2.putText(frame, f"Direction: {direction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(frame, f"Angle: {angle} deg", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
    cv2.putText(frame, f"Speed: {speed}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("Camera View", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
