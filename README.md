# 🏡 AI Dự Đoán Giá Nhà (House Price Prediction AI)
> **Mô hình học máy Linear Regression (OLS) • Backend FastAPI • UI Glassmorphism • CI/CD Tự động hóa lên Google Colab Server**

---

## 1. Tổng Quan Dự Án (Project Overview)

Dự án này là giải pháp toàn diện cho bài toán định giá bất động sản tự động sử dụng mô hình học máy **Hồi quy Tuyến tính (Linear Regression)**. Ứng dụng giải quyết bài toán:
- **Định giá chính xác**: Dự đoán giá trị căn nhà theo đơn vị Tỷ VNĐ và Triệu VNĐ/m² dựa trên 5 đặc trưng then chốt.
- **Tính minh bạch (Explainability)**: Phân rã công thức hồi quy để giải thích chính xác từng đặc trưng đóng góp bao nhiêu tiền vào tổng giá trị căn nhà (Feature Contributions).
- **Production-grade API**: Backend hiệu năng cao viết bằng FastAPI, hỗ trợ dự đoán đơn lẻ, dự đoán hàng loạt (batch), và tự động tái huấn luyện (retrain).
- **Quy trình CI/CD hoàn chỉnh**: Tự động kiểm thử unit test, đóng gói và triển khai lên máy chủ Google Colab qua mạng Tailscale Mesh (`root@100.91.70.122`) và Cloudflare HTTPS Public Tunnel.

---

## 2. Mô Hình Học Máy (Linear Regression AI Model)

### 2.1. Các đặc trưng đầu vào (Input Features)
| Đặc trưng | Tên biến | Kiểu dữ liệu | Ý nghĩa | Phạm vi chuẩn |
| :--- | :--- | :--- | :--- | :--- |
| **Diện tích sàn** | `area_m2` | `float` | Diện tích sàn sử dụng ($m^2$) | $30 - 300\ m^2$ |
| **Số phòng ngủ** | `bedrooms` | `int` | Số lượng phòng ngủ | $1 - 8$ phòng |
| **Số phòng tắm** | `bathrooms` | `int` | Số lượng phòng tắm / WC | $1 - 6$ phòng |
| **Khoảng cách** | `distance_to_center_km` | `float` | Khoảng cách tới trung tâm TP (km) | $0.5 - 30$ km |
| **Tuổi nhà** | `house_age_years` | `float` | Tuổi đời công trình xây dựng (năm) | $0 - 30$ năm |

### 2.2. Phương trình toán học hồi quy tuyến tính
$$\hat{y} = w_0 + w_1 \cdot \text{area} + w_2 \cdot \text{bedrooms} + w_3 \cdot \text{bathrooms} + w_4 \cdot \text{distance} + w_5 \cdot \text{age}$$

- $w_0$ (Intercept / Đất nền cơ bản): $\approx +0.833$ Tỷ VNĐ.
- $w_1$ (Hệ số diện tích): $\approx +0.054$ Tỷ/m² (~54 Triệu VNĐ/m²).
- $w_2$ (Hệ số phòng ngủ): $\approx +0.285$ Tỷ/phòng.
- $w_3$ (Hệ số phòng tắm): $\approx +0.201$ Tỷ/phòng.
- $w_4$ (Hệ số khoảng cách): $\approx -0.110$ Tỷ/km (càng xa trung tâm giá càng giảm).
- $w_5$ (Hệ số tuổi nhà): $\approx -0.038$ Tỷ/năm (khấu hao theo thời gian).

### 2.3. Hiệu năng mô hình (Metrics)
- **$R^2$ Score**: $\approx 0.9959$ (Mô hình giải thích được >99% phương sai dữ liệu).
- **MAE (Mean Absolute Error)**: $\approx 0.184$ Tỷ VNĐ (~184 Triệu VNĐ).
- **RMSE (Root Mean Squared Error)**: $\approx 0.230$ Tỷ VNĐ.

---

## 3. Cấu Trúc Thư Mục Chuẩn (Standard Directory Tree)

```text
/home/giangpv102/Study/Deloy_app/
├── .github/
│   └── workflows/
│       ├── ci.yml                 # Pipeline CI: Kiểm tra code, chạy pytest, test build Docker
│       └── cd.yml                 # Pipeline CD: Triển khai tự động lên Google Colab Runner
├── app/
│   ├── __init__.py
│   ├── main.py                    # Entry point FastAPI, CORS, Lifespan, Static files
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py              # API Routes: /health, /predict, /predict/batch, /model/info, /retrain
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py             # Pydantic schemas cho dữ liệu đầu vào & đầu ra
│   └── ml/
│       ├── __init__.py
│       ├── train.py               # Script sinh dữ liệu, huấn luyện mô hình và lưu artifact joblib
│       ├── predictor.py           # Engine suy luận, singleton cache, phân rã đặc trưng
│       ├── model.joblib           # File nhị phân mô hình đã được huấn luyện sẵn
│       └── metrics.json           # File metadata lưu kết quả đánh giá mô hình
├── static/
│   ├── index.html                 # Giao diện Web Glassmorphism Cyberpunk hiện đại
│   ├── style.css                  # Toàn bộ CSS phong cách Glassmorphic & micro-animations
│   └── app.js                     # Xử lý tương tác slider, gọi API real-time, biểu đồ phân rã
├── tests/
│   ├── __init__.py
│   ├── test_model.py              # Unit tests kiểm tra dữ liệu, training, logic kinh tế
│   └── test_api.py                # Integration tests kiểm tra các endpoints FastAPI
├── Dockerfile                     # Đóng gói container ứng dụng chuẩn Python 3.10
├── docker-compose.yml             # Điều phối container Docker
├── requirements.txt               # Danh sách thư viện Python cần thiết
├── .env.example                   # Biến môi trường mẫu
├── .gitignore                     # Cấu hình bỏ qua file tạm, cache, model artifacts lớn
├── run_server.sh                  # Script khởi động trực tiếp trên máy chủ Colab
├── deploy.sh                      # Script CI/CD 1-Click triển khai từ máy Local lên Colab
└── README.md
```

---

## 4. Danh Sách API (RESTful API Specifications)

| Method | Endpoint | Mô tả | Tham số mẫu |
| :--- | :--- | :--- | :--- |
| `GET` | `/` | Giao diện Web tương tác trực quan | - |
| `GET` | `/health` | Kiểm tra sức khỏe dịch vụ (CI/CD Healthcheck) | Trả về `{"status": "healthy", ...}` |
| `GET` | `/api/model/info` | Xem thông số mô hình, trọng số và metrics | Trả về danh sách weights, intercept, $R^2$, MAE |
| `POST` | `/api/predict` | Dự đoán giá cho 1 căn nhà và phân rã trọng số | `{"area_m2": 85, "bedrooms": 3, "bathrooms": 2, ...}` |
| `POST` | `/api/predict/batch` | Dự đoán hàng loạt (lên đến 100 căn) | `{"properties": [...]}` |
| `POST` | `/api/model/retrain` | Tái huấn luyện mô hình với seed mới | `{"n_samples": 1500, "random_state": 42}` |
| `GET` | `/docs` | Tài liệu API Swagger UI tương tác | Trực quan hóa OpenAPI 3.0 |

---

## 5. Hướng Dẫn Vận Hành & Triển Khai CI/CD

### Cách 1: Triển khai 1-Click bằng Script CI/CD Local (`deploy.sh`)

Script [deploy.sh](file:///home/giangpv102/Study/Deloy_app/deploy.sh) tự động thực hiện trọn vẹn quy trình CI/CD:
1. **Kiểm tra khóa SSH**: Kiểm tra file `~/.ssh/colab_ci_cd`.
2. **Kiểm tra kết nối mạng**: Ping và xác thực SSH tới máy chủ Colab `100.91.70.122`.
3. **Chạy Unit Tests cục bộ (CI)**: Tự động chạy `pytest tests/`. Nếu có bất kỳ lỗi logic nào, pipeline sẽ dừng ngay để bảo vệ server.
4. **Đồng bộ mã nguồn**: Đẩy toàn bộ source code sạch lên `/content/house_price_prediction_ai` bằng `rsync`.
5. **Kích hoạt Server**: Chạy [run_server.sh](file:///home/giangpv102/Study/Deloy_app/run_server.sh) trên Colab để dọn port 8000, tải dependencies và khởi chạy uvicorn ngầm.
6. **Kiểm tra Healthcheck**: Gọi `/health` để xác nhận server đã lên sóng.
7. **Khởi tạo Cloudflare Tunnel**: Tạo link HTTPS công khai để chia sẻ qua internet.

**Lệnh thực thi:**
```bash
cd /home/giangpv102/Study/Deloy_app
./deploy.sh 100.91.70.122
```

---

### Cách 2: Triển khai Tự Động Hóa Qua GitHub Actions (Self-Hosted Runner)

Dự án đã sẵn sàng với 2 file workflow trong `.github/workflows/`:
- [.github/workflows/ci.yml](file:///home/giangpv102/Study/Deloy_app/.github/workflows/ci.yml): Tự động chạy khi có commit/PR lên nhánh `main`, thực thi `pytest` và kiểm tra `docker build`.
- [.github/workflows/cd.yml](file:///home/giangpv102/Study/Deloy_app/.github/workflows/cd.yml): Khi được kích hoạt trên Runner `[self-hosted, colab]`, Runner sẽ tự động cập nhật code, tái khởi động ứng dụng và xuất bản link Cloudflare ra GitHub Step Summary.

---

## 6. Chạy Thử Nghiệm Tại Local (Local Development)

```bash
# 1. Cài đặt môi trường
pip install -r requirements.txt

# 2. Huấn luyện mô hình
python -m app.ml.train

# 3. Chạy unit tests
pytest -v tests/

# 4. Khởi chạy máy chủ phát triển
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Sau đó truy cập: `http://localhost:8000`.
