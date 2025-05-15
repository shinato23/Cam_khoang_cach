import cv2
import numpy as np

def detect_direction_and_speed(frame):
    frame = cv2.resize(frame, (320, 240))
    height, width = frame.shape[:2]

    # Chỉ nhìn phần sát đáy ảnh
    roi = frame[int(height * 0.8):, :]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold để phát hiện vật cản (có thể điều chỉnh nếu cần)
    _, thresh = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY_INV)

    third = width // 3
    left = thresh[:, :third]
    center = thresh[:, third:2*third]
    right = thresh[:, 2*third:]

    left_score = np.sum(left == 255)
    center_score = np.sum(center == 255)
    right_score = np.sum(right == 255)

    print(f"Scores -> Left: {left_score}, Center: {center_score}, Right: {right_score}")

    # Ngưỡng để coi vật cản là gần
    close_threshold = 3000
    margin = 500  # Độ chênh để quyết định rõ ràng trái/phải

    # Mặc định: đi thẳng
    direction = "STRAIGHT"
    steering_angle = 0
    speed = 70

    if center_score > close_threshold:
        speed = 30
        if left_score + margin < right_score:
            direction = "LEFT"
            steering_angle = -30
        elif right_score + margin < left_score:
            direction = "RIGHT"
            steering_angle = +30
        else:
            direction = "STRAIGHT"
            steering_angle = 0

    return direction, steering_angle, speed

# Dùng webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    direction, angle, speed = detect_direction_and_speed(frame)

    # Hiển thị thông tin lên màn hình
    cv2.putText(frame, f"Direction: {direction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(frame, f"Angle: {angle} deg", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
    cv2.putText(frame, f"Speed: {speed}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("Camera View", frame)
    if cv2.waitKey(1) == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
