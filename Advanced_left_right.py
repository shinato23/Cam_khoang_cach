import cv2
import numpy as np
import torch

# Load model YOLO (có thể chọn model khác tùy bạn)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def detect_direction_and_speed(frame):
    frame = cv2.resize(frame, (320, 240))
    height, width = frame.shape[:2]

    # ROI sát đáy ảnh để phát hiện vật cản
    roi = frame[int(height * 0.8):, :]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY_INV)

    # Chia ảnh thành 3 phần: Left, Center (Straight), Right
    third = width // 3
    left = thresh[:, 0:third]
    center = thresh[:, third:2*third]
    right = thresh[:, 2*third:width]

    left_score = np.sum(left == 255)
    center_score = np.sum(center == 255)
    right_score = np.sum(right == 255)

    # Ngưỡng vật cản đủ gần (threshold có thể hiệu chỉnh)
    close_threshold = 3000
    margin = 300

    direction = "STRAIGHT"
    steering_angle = 0
    speed = 70

    # Quyết định hướng dựa vào vật cản
    if center_score > close_threshold:
        speed = 30
        if right_score > left_score + margin:
            direction = "RIGHT"
            steering_angle = +30
        elif left_score > right_score + margin:
            direction = "LEFT"
            steering_angle = -30
        else:
            direction = "STRAIGHT"
            steering_angle = 0

    return direction, steering_angle, speed, frame

def estimate_distance_by_bbox_height(bbox_height):
    # Giả sử hệ số f * H_real = 1000 (bạn có thể điều chỉnh)
    if bbox_height > 0:
        return 1000 / bbox_height
    else:
        return 10000  # Rất xa

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Phát hiện hướng và tốc độ dựa vật cản
    direction, angle, speed, proc_frame = detect_direction_and_speed(frame)

    # Chạy YOLO nhận dạng vật thể trên frame gốc (không resize)
    results = model(frame)

    # Lấy kết quả detections
    detections = results.xyxy[0].cpu().numpy()  # [x1,y1,x2,y2,conf,class]

    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        conf = float(conf)
        cls = int(cls)
        label = model.names[cls]

        bbox_height = y2 - y1
        distance_cm = estimate_distance_by_bbox_height(bbox_height)

        # Chỉ nhận dạng vật thể trong khoảng cách <= 30 cm
        if conf > 0.4 and distance_cm <= 30:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
            cv2.putText(frame, f"{label} {conf:.2f} Dist:{distance_cm:.1f}cm", 
                        (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # Hiển thị thông tin điều khiển
    cv2.putText(frame, f"Direction: {direction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(frame, f"Angle: {angle} deg", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
    cv2.putText(frame, f"Speed: {speed}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    # Vẽ đường phân chia vùng Left, Center, Right
    third = frame.shape[1] // 3
    cv2.line(frame, (third, 0), (third, frame.shape[0]), (255, 0, 0), 2)
    cv2.line(frame, (2*third, 0), (2*third, frame.shape[0]), (255, 0, 0), 2)

    cv2.imshow("Camera View", frame)

    if cv2.waitKey(1) == 27:  # ESC để thoát
        break

cap.release()
cv2.destroyAllWindows()
