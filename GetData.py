import os
import pickle
import cv2
import mediapipe as mp
import numpy as np
import time

# Khoi tao MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Tao thu muc de luu du lieu da xu ly (khong luu anh)
DATA_DIR = './processed_data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Cau hinh
number_of_classes = 3  # Co the tang so luong class
samples_per_class = 100  # So luong mau cho moi class
labels_dict = {0: 'A', 1: 'B', 2: 'C'}  # Ban co the mo rong them cac cu chi

# Bien de luu tru du lieu
data = []
labels = []

# Khoi tao webcam
cap = cv2.VideoCapture(0)  # Thu 0 neu 1 khong hoat dong


# Ham trich xuat dac trung tu khung hinh
def extract_hand_features(frame):
    # Chuyen doi sang RGB de xu ly voi MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Phat hien ban tay
    results = hands.process(frame_rgb)

    feature_vector = []
    hand_detected = False

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Ve cac diem moc ban tay len khung hinh
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Trich xuat toa do
            x_ = []
            y_ = []

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            # Chuan hoa toa do
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                feature_vector.append(x - min(x_))
                feature_vector.append(y - min(y_))

            hand_detected = True
            break  # Chi xu ly ban tay dau tien duoc phat hien

    return feature_vector, hand_detected, frame


# Thu thap du lieu cho tung lop
print("Bat dau thu thap du lieu...")
for class_idx in range(number_of_classes):
    print(f'===== Thu thap du lieu cho cu chi {labels_dict[class_idx]} (Lop {class_idx}) =====')

    # Doi nguoi dung san sang
    print('Hien thi camera. Nhan "Q" khi ban san sang thu thap du lieu cho cu chi nay')
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Khong the doc tu webcam!")
            break

        # Phat hien va ve diem moc ban tay ngay ca trong giai doan chuan bi
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

        # Hien thi huong dan
        cv2.putText(
            frame,
            f'Chuan bi cho cu chi {labels_dict[class_idx]}? Nhan "Q" de bat dau!',
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        cv2.imshow('Thu thap du lieu', frame)

        # Thoat khi nhan 'q'
        if cv2.waitKey(25) == ord('q'):
            break

    print(f'Bat dau thu thap {samples_per_class} mau...')
    counter = 0

    # Doi mot chut de nguoi dung chuan bi
    time.sleep(2)

    # Thu thap mau
    while counter < samples_per_class:
        ret, frame = cap.read()
        if not ret:
            continue

        # Trich xuat dac trung
        features, hand_detected, annotated_frame = extract_hand_features(frame)

        # Hien thi trang thai
        cv2.putText(
            annotated_frame,
            f'Cu chi {labels_dict[class_idx]}: {counter}/{samples_per_class}',
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        cv2.imshow('Thu thap du lieu', annotated_frame)
        cv2.waitKey(1)

        # Neu phat hien duoc tay, luu dac trung
        if hand_detected and len(features) > 0:
            data.append(features)
            labels.append(class_idx)
            counter += 1

            # Hien thi tien trinh
            if counter % 10 == 0:
                print(f'Da thu thap {counter}/{samples_per_class} mau')

            # Tam dung mot chut de dam bao su da dang cua du lieu
            time.sleep(0.1)

# Luu du lieu da xu ly vao file pickle
print("Da thu thap xong! Luu du lieu...")
data_file = os.path.join(DATA_DIR, 'data.pickle')
with open(data_file, 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print(f"Du lieu da duoc luu vao {data_file}")
print(f"Tong so mau da thu thap: {len(data)}")

# Giai phong tai nguyen
cap.release()
cv2.destroyAllWindows()