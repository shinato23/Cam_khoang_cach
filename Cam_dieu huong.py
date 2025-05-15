import cv2
import numpy as np

def detect_direction_and_speed(frame):
    # Resize ảnh cho đơn giản
    frame = cv2.resize(frame, (320, 240))
    height, width = frame.shape[:2]

    # Cắt vùng quan tâm phía dưới ảnh (nơi sát mặt đất)
    roi = frame[int(height * 0.6):, :]

    # Chuyển sang ảnh xám để xử lý đơn giản
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Phát hiện vật cản bằng phân ngưỡng
    _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)

    # Chia ảnh thành 3 vùng: trái - giữa - phải
    third = width // 3
    left = thresh[:, :third]
    center = thresh[:, third:2*third]
    right = thresh[:, 2*third:]

    # Đếm số điểm trắng (nghi ngờ là vật cản)
    left_score = np.sum(left == 255)
    center_score = np.sum(center == 255)
    right_score = np.sum(right == 255)

    # Ngưỡng giả lập "vật cản gần"
    close_threshold = 1000  # bạn có thể điều chỉnh ngưỡng này tùy thực tế

    print(f"Scores -> Left: {left_score}, Center: {center_score}, Right: {right_score}")

    # Mặc định đi thẳng
    direction = "STRAIGHT"
    steering_angle = 0

    # Nếu có vật cản gần ở giữa thì mới xét tránh né
    if center_score > close_threshold:
        if left_score > right_score and left_score > center_score:
            direction = "RIGHT"
            steering_angle = +30
        elif right_score > left_score and right_score > center_score:
            direction = "LEFT"
            steering_angle = -30
        else:
            direction = "STRAIGHT"
            steering_angle = 0

        speed = 30  # Giảm tốc độ khi gần vật
    else:
        speed = 70  # Tốc độ bình thường

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
    if cv2.waitKey(1) == 27:  # ESC để thoát
        break

cap.release()
cv2.destroyAllWindows()
