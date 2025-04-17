import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import time

# Bat dau do thoi gian
start_time = time.time()

print("Dang tai du lieu...")
try:
    # Tai du lieu tu file pickle
    data_path = './processed_data/data.pickle'
    if not os.path.exists(data_path):
        data_path = './data.pickle'  # Duong dan thay the neu khong tim thay file

    data_dict = pickle.load(open(data_path, 'rb'))

    # Chuyen doi du lieu va nhan thanh mang numpy
    data = np.asarray(data_dict['data'])
    labels = np.asarray(data_dict['labels'])

    print(f"Da tai xong {len(data)} mau du lieu voi {len(set(labels))} lop.")

    # Kiem tra du lieu co kich thuoc dong nhat
    data_shapes = [len(d) for d in data]
    if len(set(data_shapes)) > 1:
        print(f"Canh bao: Du lieu co kich thuoc khong dong nhat. Dang sua chua...")
        # Tim kich thuoc pho bien nhat
        max_len = max(data_shapes)
        # Chuan hoa kich thuoc
        for i in range(len(data)):
            if len(data[i]) < max_len:
                # Them 0 vao cuoi
                data[i] = np.pad(data[i], (0, max_len - len(data[i])), 'constant')
        data = np.asarray(data)
        print(f"Da sua chua xong. Tat ca mau co kich thuoc {max_len}.")

    # Chuan hoa du lieu
    print("Dang chuan hoa du lieu...")
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Chia du lieu thanh tap huan luyen va tap kiem tra
    print("Chia du lieu thanh tap huan luyen va tap kiem tra...")
    x_train, x_test, y_train, y_test = train_test_split(
        data_scaled, labels, test_size=0.2, shuffle=True, stratify=labels, random_state=42
    )

    print(f"So luong mau huan luyen: {len(x_train)}")
    print(f"So luong mau kiem tra: {len(x_test)}")

    # Kiem tra phan phoi lop
    unique_labels, counts = np.unique(y_train, return_counts=True)
    print("Phan phoi lop trong tap huan luyen:")
    for label, count in zip(unique_labels, counts):
        print(f"  Lop {label}: {count} mau")

    # 1. Tim kiem luoi tham so toi uu
    print("\nDang tim kiem tham so toi uu cho mo hinh RandomForest...")
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Su dung GridSearchCV de tim tham so toi uu
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1  # Su dung tat ca CPU co san
    )

    grid_search.fit(x_train, y_train)

    print(f"Tham so tot nhat: {grid_search.best_params_}")
    print(f"Do chinh xac tot nhat trong qua trinh tim kiem: {grid_search.best_score_:.4f}")

    # 2. Huan luyen mo hinh voi tham so tot nhat
    print("\nDang huan luyen mo hinh voi tham so tot nhat...")
    best_model = grid_search.best_estimator_

    # 3. Danh gia mo hinh
    print("\nDanh gia mo hinh:")
    y_predict = best_model.predict(x_test)
    accuracy = accuracy_score(y_test, y_predict)
    print(f"Do chinh xac tren tap kiem tra: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    # Bao cao chi tiet
    print("\nBao cao phan loai chi tiet:")
    print(classification_report(y_test, y_predict))

    # Tinh va hien thi ma tran nham lan
    conf_matrix = confusion_matrix(y_test, y_predict)
    print("Ma tran nham lan:")
    print(conf_matrix)

    # Ve bieu do ma tran nham lan
    plt.figure(figsize=(10, 7))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Ma tran nham lan')
    plt.colorbar()

    # Dat nhan
    classes = sorted(np.unique(labels))
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    # Them so lieu
    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, conf_matrix[i, j],
                     horizontalalignment="center",
                     color="white" if conf_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Nhan thuc te')
    plt.xlabel('Nhan du doan')

    # Tao thu muc cho mo hinh va bieu do
    models_dir = './models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    # Luu bieu do
    plt.savefig(os.path.join(models_dir, 'confusion_matrix.png'))

    # 4. Luu mo hinh va cac thong tin khac
    print("\nDang luu mo hinh va cac thong tin can thiet...")

    # Luu mo hinh
    model_file = os.path.join(models_dir, 'model.p')
    with open(model_file, 'wb') as f:
        pickle.dump({'model': best_model}, f)

    # Luu bo chuan hoa
    scaler_file = os.path.join(models_dir, 'scaler.joblib')
    joblib.dump(scaler, scaler_file)

    # Luu thong tin bo sung
    info = {
        'num_classes': len(np.unique(labels)),
        'accuracy': accuracy,
        'parameters': grid_search.best_params_,
        'feature_importance': best_model.feature_importances_
    }

    info_file = os.path.join(models_dir, 'model_info.p')
    with open(info_file, 'wb') as f:
        pickle.dump(info, f)

    # Tinh tong thoi gian chay
    end_time = time.time()
    training_time = end_time - start_time

    print(f"\nQua trinh huan luyen hoan tat trong {training_time:.2f} giay!")
    print(f"Mo hinh da duoc luu tai: {model_file}")
    print(f"Bo chuan hoa da duoc luu tai: {scaler_file}")
    print(f"Bieu do ma tran nham lan da duoc luu tai: {os.path.join(models_dir, 'confusion_matrix.png')}")

except Exception as e:
    print(f"Loi: {str(e)}")