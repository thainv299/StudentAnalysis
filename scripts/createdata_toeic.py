import pandas as pd
import numpy as np
import random
import math
import os
num_students = 20   
output_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "input", "data_demo_1_toeic.csv")

# Mapping mã môn -> tên môn (3 ngành: CNTT, CNPM, Truyền thông & Mạng)
SUBJECT_NAMES = {
    # === Môn đại cương / chung ===
    '18124': 'Toán cao cấp',
    '29101': 'Kỹ năng mềm 1',
    # === Môn cơ sở ngành CNTT ===
    '17102': 'Tin học văn phòng',
    '17104': 'Tin học đại cương',
    '17206': 'Kỹ thuật lập trình C',
    '17232': 'Toán rời rạc',
    '17233': 'Cấu trúc dữ liệu và giải thuật',
    '17236': 'Lập trình hướng đối tượng',
    '17302': 'Kiến trúc máy tính và TBNV',
    '17301': 'Kỹ thuật vi xử lý',
    '17303': 'Nguyên lý hệ điều hành',
    '17426': 'Cơ sở dữ liệu',
    '17506': 'Mạng máy tính',
    # === Môn chuyên ngành ===
    '17212': 'An toàn và bảo mật thông tin',
    '17221': 'Xử lý ảnh',
    '17226': 'Thị giác máy tính',
    '17234': 'Trí tuệ nhân tạo',
    '17314': 'Phát triển ứng dụng mã nguồn mở',
    '17332': 'Công nghệ Internet of Things',
    '17335': 'Lập trình Windows',
    '17340': 'Phát triển ứng dụng trên nền web',
    '17423': 'Lập trình thiết bị di động',
    '17427': 'Phân tích và thiết kế hệ thống',
    '17432': 'Nhập môn công nghệ phần mềm',
    '17434': 'Phát triển ứng dụng với CSDL',
    '17523': 'Java cơ bản',
    '17542': 'Tiếp thị trực tuyến',
}

subjects = list(SUBJECT_NAMES.keys())

data = []

for i in range(num_students):

    ma_sv = f"SV{i:05d}"
    ho_ten = f"SinhVien_{i}"

    row = {
        "ma_sv": ma_sv,
        "ho_ten": ho_ten
    }
    group = random.choices(
        ["xuat_sac", "kha", "trung_binh", "nguy_hiem"],
        weights=[0.2, 0.3, 0.3, 0.2]
    )[0]

    for subject in subjects:

        if group == "xuat_sac":
            score = np.random.normal(8.8, 0.4)
        elif group == "kha":
            score = np.random.normal(7.5, 0.6)
        elif group == "trung_binh":
            score = np.random.normal(6.0, 0.8)
        else:  # nguy_hiem
            # 40% khả năng trượt môn
            if random.random() < 0.4:
                score = np.random.normal(3.5, 0.5)
            else:
                score = np.random.normal(5.0, 0.8)

        score = max(0, min(10, score))
        score = round(score, 1)

        row[subject] = score

    # Sinh điểm TOEIC dựa trên Group
    if group == "xuat_sac":
        toeic = round(np.random.normal(700, 100))
    elif group == "kha":
        toeic = round(np.random.normal(550, 80))
    elif group == "trung_binh":
        toeic = round(np.random.normal(450, 70))
    else:
        toeic = round(np.random.normal(350, 60))
        
    # Làm tròn để giống điểm thật (chia hết cho 5)
    toeic = max(200, min(990, int(round(toeic / 5.0) * 5.0)))
    row["TOEIC"] = toeic

    data.append(row)

df_demo = pd.DataFrame(data)
df_demo.to_csv(output_file, sep=";", index=False)
print("Da tao xong file:", output_file)
print("Kich thuoc du lieu:", df_demo.shape)
print(df_demo.head())