# 🛡️ AntiGravity Transaction Anomaly Radar

Ứng dụng Web phát hiện giao dịch bất thường (Anomaly Detection) được xây dựng trên nền tảng **Streamlit** và các mô hình Học máy (**Logistic Regression, Decision Tree, Random Forest**) từ thư viện Scikit-Learn. Hệ thống hỗ trợ phân tích trực quan luồng giao dịch, huấn luyện mô hình dự đoán và quét kiểm tra giao dịch thời gian thực.

## 🚀 Tính năng nổi bật

1. **Dashboard Tổng quan (Overview):**
   * Hiển thị biểu đồ Plotly tương tác: Tần suất giao dịch theo ngày, tỷ lệ bất thường theo chi nhánh, phân bố giao dịch theo kênh và theo giờ trong ngày.
   * Cung cấp các thẻ KPIs trực quan về tổng số giao dịch, tỷ lệ giao dịch bất thường, và tổng giá trị nghi ngờ (VND).
2. **Huấn luyện & Đánh giá mô hình:**
   * Tự động tiền xử lý dữ liệu và trích xuất đặc trưng (Feature Engineering) từ tệp dữ liệu thô `transactions_Q1_demo.csv`.
   * So sánh trực quan hiệu năng của 3 mô hình chính: **Logistic Regression, Decision Tree, Random Forest** dựa trên các chỉ số Accuracy, Precision, Recall, F1-Score.
   * Trực quan hóa Ma trận nhầm lẫn (Confusion Matrix) và Đánh giá Độ quan trọng đặc trưng (Feature Importance).
3. **Phát hiện & Kiểm tra giao dịch (Real-time & Batch):**
   * **Kiểm tra đơn lẻ (Ad-hoc check):** Cho phép nhập tay thông tin giao dịch để quét kiểm tra ngay lập tức với xác suất rủi ro (%) cụ thể.
   * **Quét hàng loạt (Batch Scan):** Cho phép tải lên tệp CSV/Excel giao dịch mới để quét, hiển thị danh sách cảnh báo bất thường và tải báo cáo định dạng CSV.

---

## 🛠️ Hướng dẫn cài đặt và Chạy cục bộ (Local Run)

Để chạy ứng dụng trên máy tính của bạn, hãy thực hiện theo các bước sau:

### Bước 1: Cài đặt Python
Tải xuống và cài đặt **Python (phiên bản 3.9 - 3.12)** từ trang chủ [python.org](https://www.python.org/downloads/). Đảm bảo tích chọn mục **"Add Python to PATH"** trong quá trình cài đặt.

### Bước 2: Tạo Môi trường ảo & Cài đặt Thư viện
Mở Command Prompt (cmd) hoặc PowerShell, di chuyển đến thư mục của dự án và chạy các lệnh sau:

```bash
# Tạo môi trường ảo
python -m venv .venv

# Kích hoạt môi trường ảo
# Trên Windows:
.venv\Scripts\activate
# Trên macOS/Linux:
source .venv/bin/activate

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

### Bước 3: Khởi chạy ứng dụng Streamlit
Chạy lệnh sau để khởi động máy chủ cục bộ:
```bash
streamlit run app.py
```
Trình duyệt web sẽ tự động mở trang ứng dụng tại địa chỉ mặc định: `http://localhost:8501`.

---

## 🌐 Hướng dẫn Deploy lên Streamlit Cloud

Streamlit cung cấp nền tảng **Streamlit Community Cloud** miễn phí để bạn chia sẻ ứng dụng của mình lên Internet.

### Bước 1: Đẩy mã nguồn lên GitHub
1. Tạo một Repository mới trên GitHub của bạn.
2. Đẩy các tệp sau lên GitHub:
   * `app.py` (Mã nguồn ứng dụng)
   * `requirements.txt` (Danh sách thư viện phụ thuộc)
   * `transactions_Q1_demo.csv` (File dữ liệu mẫu để chạy ứng dụng mặc định)
   * `README.md` (Hướng dẫn này)

### Bước 2: Deploy ứng dụng
1. Truy cập trang [share.streamlit.io](https://share.streamlit.io/) và đăng nhập bằng tài khoản GitHub của bạn.
2. Nhấn nút **"Create app"** ở góc trên cùng bên phải.
3. Chọn Repository, Branch (thường là `main` hoặc `master`), và chỉ định tệp chạy chính là `app.py`.
4. Nhấn **"Deploy!"**. Quá trình cài đặt môi trường và khởi chạy sẽ mất từ 1-3 phút. Sau khi hoàn thành, bạn sẽ nhận được một đường link URL công khai để chia sẻ ứng dụng.
