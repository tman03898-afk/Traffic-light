# Hệ Thống Phát Hiện Vi Phạm Giao Thông (Camera Fisheye) - YOLO11

Hệ thống cung cấp quy trình sử dụng YOLO11 để nhận diện và ghi nhận phương tiện vượt đèn đỏ. Project này được tinh chỉnh cho dữ liệu camera góc rộng (Fisheye) trên thực tế.

## 🚀 Các Tính Năng
- **Quét nhiều đèn tín hiệu**: Hỗ trợ định nghĩa nhiều vùng quét đèn (Light ROIs) cùng lúc để phòng trường hợp đèn bị phương tiện che khuất.
- **Xử lý vi phạm theo vùng (Zone-based)**: So sánh đối tượng đi qua vùng chờ (Zone 1) và vùng vi phạm (Zone 2) với trạng thái đèn để xác thực vi phạm.
- **Huấn luyện mô hình**: Cung cấp script để huấn luyện đồng nhất cho các model YOLO.
- **Lưu file kết quả**: Tự động tính checksum SHA256 và lưu riêng ảnh crop vi phạm, ảnh toàn cảnh và đoạn video ngắn.
- **Công cụ hỗ trợ GUI**: Tool giao diện dùng chuột hỗ trợ lấy tọa độ ROI nhanh chóng cho bất kỳ luồng camera nào.

---

## 🏗️ Các Script Chính

Các thành phần cốt lõi của hệ thống bao gồm 3 file chính:

### 1. Hub huấn luyện (`scripts/research/train_yolo.py`)
Script tổng hợp dùng để huấn luyện mô hình YOLO11 với các hệ số (hyperparameters) cấu hình sẵn.
```powershell
python scripts/research/train_yolo.py --preset optimized_legacy --data configs/data.yaml
```

### 2. Công cụ lấy toạ độ ROI (`scripts/utils/get_polygon_roi.py`)
Giúp người dùng vẽ và xác định tọa độ vùng xử lý vi phạm bằng chuột.
```powershell
python scripts/utils/get_polygon_roi.py --source detect.mp4
```
*   **Zone 1**: Vùng chờ trước vạch.
*   **Zone 2**: Vùng chạy vi phạm.
*   **Zone 3+**: Các vùng quét đèn.

### 3. File nhận diện luồng (`scripts/detect.py`)
File thực thi dùng để chạy logic mô hình nhận diện xe và xử lý phạt nguội trên video phân tích.
```powershell
python scripts/detect.py --source "detect.mp4" --light-weights "models/best_collection/yolo11n_v2.pt" --roi1 "..." --roi2 "..." --light-roi "roi_1" "roi_2"
```

---

## 📊 Đánh Giá Benchmark
- Dùng `scripts/utils/generate_leaderboard.py` để thống kê Precision/Recall/mAP của nhiều mô hình đã huấn luyện.
- Dùng `scripts/utils/infer_video.py` để đo FPS, benchmark phần cứng của hệ thống đo.

## 🛠️ Cài Đặt
1.  Clone repository này về máy.
2.  Cài đặt các thư viện cần thiết:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Tải Dataset:**
    Do giới hạn về dung lượng, Dataset chưa được đẩy trực tiếp lên GitHub. Hãy tải file Dataset về và giải nén vào thư mục độc lập với source code (ví dụ: `../Data/`).
    *   [🔗 https://drive.google.com/file/d/1L-Ul_frmtA8UrXKJ4Oy2PWp4vRcr4ojk/view]()

---

## 📂 Cấu Trúc File
- `scripts/`: Source code Python chính của project.
- `configs/`: File YAML tham chiếu Dataset.
- `models/best_collection/`: Nơi chứa file model pre-trained hoàn thiện nhất.
- `outputs/`: Nơi xuất log lỗi vi phạm xử lý sau khi chạy.

---
*Dành cho các hệ thống giám sát và xử lý vi phạm giao thông tự động.*
