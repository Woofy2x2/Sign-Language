
# Hệ thống nhận diện ký hiệu tay bằng MediaPipe & Random Forest

Dự án này bao gồm ba bước chính: thu thập dữ liệu ký hiệu tay, huấn luyện mô hình phân loại và sử dụng mô hình để dự đoán ký hiệu tay theo thời gian thực.
## Tham khảo / Reference

Dự án này được lấy cảm hứng từ Computer vision engineer:

🔗 https://github.com/computervisioneng/sign-language-detector-python

## Cấu trúc dự án

```
├── GetData.py                # Thu thập và trích xuất đặc trưng ký hiệu tay
├── TrainRandomForest.py      # Huấn luyện mô hình Random Forest
├── InferenceClassifier.py    # Dự đoán ký hiệu tay theo thời gian thực
├── processed_data/           # Thư mục chứa dữ liệu đã xử lý
└── models/                   # Thư mục chứa mô hình huấn luyện và scaler
```

---

## 1. Thu thập dữ liệu - `GetData.py`

### Mục đích:
- Sử dụng webcam và thư viện MediaPipe để phát hiện bàn tay và trích xuất đặc trưng (tọa độ các điểm mốc).
- Gán nhãn thủ công cho các ký hiệu (A, B, C, ...).
- Lưu dữ liệu đặc trưng và nhãn dưới dạng file `data.pickle`.

### Cách dùng:
```bash
python GetData.py
```

### Ghi chú:
- Mỗi lần thu thập bạn sẽ chọn một lớp ký hiệu tay, hệ thống sẽ hiển thị camera và yêu cầu bạn nhấn `Q` để bắt đầu.
- Mặc định thu thập 100 mẫu cho mỗi lớp (có thể thay đổi trong mã).
- Dữ liệu được lưu tại `./processed_data/data.pickle`.

---

## 2. Huấn luyện mô hình - `TrainRandomForest.py`

### Mục đích:
- Tải và chuẩn hóa dữ liệu đã thu thập.
- Huấn luyện mô hình `RandomForestClassifier` với `GridSearchCV` để chọn siêu tham số tốt nhất.
- Lưu mô hình, bộ chuẩn hóa và thông tin mô hình (accuracy, importance, ...) vào thư mục `models/`.

### Cách dùng:
```bash
python TrainRandomForest.py
```

### Kết quả đầu ra:
- `models/model.p`: Mô hình đã huấn luyện.
- `models/scaler.joblib`: Bộ chuẩn hóa dữ liệu.
- `models/model_info.p`: Thông tin mô hình.
- `models/confusion_matrix.png`: Ma trận nhầm lẫn mô hình.

---

## 3. Dự đoán thời gian thực - `InferenceClassifier.py`

### Mục đích:
- Tải mô hình và scaler đã lưu.
- Mở webcam để nhận diện ký hiệu tay trong thời gian thực.
- Hiển thị kết quả dự đoán (A, B, C) trên khung hình.

### Cách dùng:
```bash
python InferenceClassifier.py
```

### Ghi chú:
- Nhấn `Q` để thoát khỏi chương trình.
- Cần đảm bảo các file mô hình (`model.p`, `scaler.joblib`) có sẵn trong thư mục `models/`.

---

## Yêu cầu thư viện

Cài đặt các thư viện cần thiết:
```bash
pip install opencv-python mediapipe numpy scikit-learn matplotlib joblib
```

---

## Ghi chú bổ sung

- Bạn có thể mở rộng số lượng ký hiệu bằng cách thay đổi `labels_dict` và `number_of_classes` trong `GetData.py`.
- Đảm bảo điều kiện ánh sáng và góc tay ổn định để thu thập dữ liệu và nhận diện chính xác hơn.


---


# Hand Sign Recognition System using MediaPipe & Random Forest

This project includes three main steps: collecting hand sign data, training a classification model, and using the model for real-time sign recognition.

## Project Structure

```
├── GetData.py                # Collect and extract hand sign features
├── TrainRandomForest.py      # Train Random Forest model
├── InferenceClassifier.py    # Real-time hand sign prediction
├── processed_data/           # Folder for processed data
└── models/                   # Folder for trained models and scaler
```

---

## 1. Data Collection - `GetData.py`

### Purpose:
- Use webcam and MediaPipe to detect hands and extract keypoint features.
- Manually label each sign (e.g., A, B, C).
- Save processed features and labels into `data.pickle`.

### Usage:
```bash
python GetData.py
```

### Notes:
- For each class, the system shows the camera and asks you to press `Q` to start collecting samples.
- By default, 100 samples are collected per class (you can modify this in the script).
- Data will be saved in `./processed_data/data.pickle`.

---

## 2. Model Training - `TrainRandomForest.py`

### Purpose:
- Load and normalize collected data.
- Train a `RandomForestClassifier` using `GridSearchCV` to find the best hyperparameters.
- Save the model, scaler, and extra info (accuracy, importance, etc.) into the `models/` folder.

### Usage:
```bash
python TrainRandomForest.py
```

### Outputs:
- `models/model.p`: Trained model.
- `models/scaler.joblib`: Scaler for input normalization.
- `models/model_info.p`: Model metadata.
- `models/confusion_matrix.png`: Confusion matrix visualization.

---

## 3. Real-time Inference - `InferenceClassifier.py`

### Purpose:
- Load the trained model and scaler.
- Use webcam to detect and recognize hand signs in real time.
- Display predicted label (A, B, C) on the screen.

### Usage:
```bash
python InferenceClassifier.py
```

### Notes:
- Press `Q` to exit.
- Make sure model files (`model.p`, `scaler.joblib`) are available in `models/`.

---

## Required Libraries

Install dependencies:
```bash
pip install opencv-python mediapipe numpy scikit-learn matplotlib joblib
```

---

## Additional Notes

- You can expand the number of signs by editing `labels_dict` and `number_of_classes` in `GetData.py`.
- Ensure good lighting and hand posture for better accuracy during both data collection and inference.
