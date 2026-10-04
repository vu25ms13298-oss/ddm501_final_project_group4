# 🎮 LPR Traffic & Drift Simulation Toolkit

Bộ công cụ giả lập lưu lượng truy cập thực tế và thử nghiệm trôi dạt dữ liệu (Data & Prediction Drift) cho hệ thống nhận diện biển số xe Vietnamese LPR.

---

## 🎯 Mục đích

1. **Tạo tải thực tế (Live Traffic)**: Gửi các request nhận diện biển số lên API đang chạy với tốc độ RPS và độ đồng thời kiểm soát được.
2. **Kích hoạt trôi dạt dữ liệu (Drift Injection)**: Mô phỏng ánh sáng tối (dusk/night), camera bị mờ do mưa, thay đổi độ phân giải, camera bị lệch góc hoặc biển số bị che lấp.
3. **Mô phỏng trôi dạt kết quả (Prediction Drift)**: Bắn các ảnh bối cảnh không có biển số để quan sát tỷ lệ định dạng biển số hợp lệ tụt dốc.
4. **Kiểm tra Rate Limiting**: Bắn burst traffic vượt ngưỡng 60 req/phút để kích hoạt phản hồi HTTP 429 và alert `RateLimitBurst`.
5. **Quan sát trực quan**: Đồ thị trên Grafana (`http://localhost:3000`) và danh sách cảnh báo trên Prometheus (`http://localhost:9090/alerts`) thay đổi ngay lập tức theo thời gian thực.

---

## 📦 Cài đặt

```bash
cd simulations
pip install -r requirements.txt
```

---

## 🚀 Hướng dẫn nhanh

### 1. Kiểm tra kết nối dịch vụ
```bash
# Trên Windows PowerShell:
.\quick_test.ps1

# Trên Linux / macOS:
./quick_test.sh
```

### 2. Chạy kịch bản cơ bản qua CLI
```bash
# 60 request bình thường (tốc độ an toàn 0.8 req/s tránh dính rate limit):
python run_simulation.py -n 60 -s normal

# Thử nghiệm ban đêm (nightfall drift):
python run_simulation.py -n 50 -s night

# Cố tình bắn nhanh vượt rate limit để quan sát lỗi 429:
python run_simulation.py -n 70 --rps 10 --concurrency 4 --no-respect-rate-limit

# Gửi 15 request lỗi (file rỗng, payload quá lớn) để kích hoạt counter lỗi:
python run_simulation.py --bad-requests 15
```

---

## 📋 7 Kịch bản định sẵn (`scenarios.py`)

Chạy kịch bản bằng số:
```bash
python scenarios.py <số_kịch_bản>
```

| # | Tên Kịch Bản | Mô tả chi tiết | Hiện tượng quan sát được trên Grafana & Prometheus |
|---|---|---|---|
| **1** | **Normal Day Traffic** | 100 requests biển số rõ nét ban ngày | RPS ~0.8, latency p95 < 400ms, tỷ lệ thành công ~100%, không alert nào kích hoạt. |
| **2** | **Nightfall (Gradual Drift)** | Chuyển dần: Normal (40) $\rightarrow$ Dusk (40) $\rightarrow$ Night (50) | Panel *Input Brightness (5m vs 1h)* tụt dốc; alert `InputBrightnessDriftFast` chuyển sang **FIRING**. |
| **3** | **Camera Swap (Sudden Shift)** | Đột ngột đổi sang camera độ phân giải thấp + nén ảnh mạnh | Panel *Input Width (5m vs 1h)* giảm mạnh; alert `InputResolutionDriftFast` chuyển sang **FIRING**. |
| **4** | **Adverse Weather Mix** | Xen kẽ trời mưa (motion blur) và bùn đất che biển số | Điểm số format và số lượng ký tự nhận diện được dao động mạnh. |
| **5** | **Prediction Drift** | Gửi ảnh phong cảnh/nhiễu không chứa biển số xe | Panel *Recognition Success Rate (5m)* tụt xuống dưới 50%; alert `LowRecognitionSuccessRateFast` kích hoạt. |
| **6** | **Traffic Spike & Rate Limit** | Bắn dồn dập 70 requests trong vài giây không chờ backoff | Panel */predict HTTP Status Codes* xuất hiện vạch đỏ HTTP 429; alert `RateLimitBurst` kích hoạt. |
| **7** | **Bad Input Telemetry** | Gửi file rỗng, file hỏng, file quá 10MB | Panel *Prediction Errors* ghi nhận các loại lỗi `empty_file`, `invalid_image`, `payload_too_large`. |

Chạy toàn bộ 7 kịch bản tuần tự:
```bash
python scenarios.py
```

---

## ⚙️ Cấu hình (`config.yaml`)

File [`config.yaml`](config.yaml) cho phép tùy chỉnh:
- `api.base_url`: Địa chỉ API (mặc định `http://localhost:8000`).
- `defaults.rps`: Tốc độ bắn request (mặc định `0.8` req/s để nằm trong rate limit 60/phút).
- `scenarios.<name>.transforms`: Tham số chỉnh sửa độ sáng (`brightness`), độ nhiễu (`noise`), làm mờ (`motion_blur`), thu nhỏ (`downscale`), xoay nghiêng (`rotate`), che lấp (`occlusion`).
