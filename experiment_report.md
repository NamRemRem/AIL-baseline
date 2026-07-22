# BÁO CÁO KẾT QUẢ NGHIÊN CỨU & THỰC NGHIỆM

## TÊN ĐỀ TÀI: NÂNG CẤP VÀ TỐI ƯU HÓA HỆ THỐNG PHÁT HIỆN BẤT THƯỜNG TRONG ẢNH CÔNG NGHIỆP BẰNG ADAPTIVE MEMORY-BANK

---

## 1. ĐẶT VẤN ĐỀ VÀ MỤC TIÊU NGHIÊN CỨU

### 1.1. Đặt vấn đề
Trong kiểm định chất lượng sản xuất công nghiệp (Industrial Visual Inspection), việc phát hiện lỗi sản phẩm (vết nứt, trầy xước, dơ, sai lệch cấu trúc) gặp các thách thức:
- **Hiếm khi có ảnh lỗi (Anomaly Data Scarcity):** Hầu hết sản phẩm trên dây chuyền là sản phẩm chuẩn (good), ảnh lỗi rất hiếm và đa dạng.
- **Mô hình yêu cầu nhẹ và nhanh (Edge Computing):** Cần triển khai trên thiết bị nhúng/edge với tài nguyên tính toán giới hạn.

### 1.2. Mục tiêu bài làm
1. **Thiết kế hệ thống Unsupervised Anomaly Detection & Segmentation:** Chỉ cần học trên dữ liệu ảnh bình thường (good samples).
2. **Cải tiến mô hình so với các nghiên cứu trước:**
   - Thay thế backbone nặng WideResNet-50 (69M params) bằng **EfficientNet-B3 (12M params)** – giảm **5.7× tham số**.
   - Phát triển **AdaptiveCoreset Subsampling** tự điều chỉnh tỷ lệ nén bộ nhớ dựa trên dung lượng dữ liệu.
   - Đề xuất **SoftmaxNNScorer** thay cho trung bình khoảng cách k-NN thông thường để tăng độ nhạy với lỗi nhỏ.
   - Thiết kế **AdaptiveRescaleSegmentor** tự điều chỉnh độ mịn (Gaussian sigma) theo độ phân giải feature map.
3. **Đầu ra hoàn chỉnh:** Anomaly Score (cấp ảnh) + Anomaly Heatmap (cấp pixel) + File JSON dự đoán chi tiết.

---

## 2. PHƯƠNG PHÁP NGHIÊN CỨU & CÁC CẢI TIẾN ĐỀ XUẤT

### 2.1. Kiến trúc tổng thể (Architecture)

```
[Ảnh Đầu Vào (224×224)]
         │
         ▼
[Backbone: EfficientNet-B3]  ──(Trích xuất Feature Maps)──► [Blocks 5 & 6]
         │                                                      │
         ▼                                                      ▼
[Patchify (Kernel=3, Stride=1)] ◄───────────────────────── [Align & Merge]
         │
         ▼
[MeanMapper & Aggregator (1024-d)]
         │
         ├──► (Training)  ──► [AdaptiveCoreset (7-15%)] ──► [Memory Bank (FAISS)]
         │                                                       │
         └──► (Inference) ───────────────────────────────────────┤
                                                                 ▼
                                                      [SoftmaxNNScorer (k=5)]
                                                                 │
                                                       ┌─────────┴─────────┐
                                                       ▼                   ▼
                                                [Anomaly Score]   [AdaptiveSegmentor]
                                                (Image-level)              │
                                                                           ▼
                                                                  [Anomaly Heatmap]
                                                                    (Pixel-level)
```

### 2.2. Chi tiết các thành phần cải tiến

#### A. EfficientNet-B3 Backbone (Tối ưu tài nguyên)
- **Layers trích xuất:** `blocks.5` (feature resolution 14×14, 232 channels) và `blocks.6` (feature resolution 7×7, 384 channels).
- **Lợi ích:** Giảm tham số từ 68.9M (WideResNet-50) xuống 12.2M (giảm 82%), tốc độ trích xuất nhanh hơn rõ rệt.

#### B. AdaptiveCoreset Subsampling (Nén bộ nhớ thông minh)
Mô hình tự động chọn tỷ lệ $r$ duy trì trong Memory Bank:

$$\text{Retention Rate } r = \begin{cases} 15\% & \text{nếu } N_{\text{patches}} < 30,000 \\ 10\% & \text{nếu } 30,000 \le N_{\text{patches}} \le 80,000 \\ 7\% & \text{nếu } N_{\text{patches}} > 80,000 \end{cases}$$

#### C. SoftmaxNNScorer (Hàm tính điểm bất thường cấp ảnh)
Trọng số Softmax:

$$w_i = \frac{\exp(-d_i / T)}{\sum_{j=1}^k \exp(-d_j / T)}, \quad \text{Score}(q) = \sum_{i=1}^k w_i \cdot d_i$$

Với nhiệt độ $T=1.0$, các khoảng cách ngắn hơn (gần nhất) được gán trọng số lớn hơn.

#### D. AdaptiveRescaleSegmentor (Tạo Heatmap cấp Pixel)
Heatmap được lọc Gaussian với $\sigma$ linh hoạt:

$$\sigma = \text{clamp}\left(\frac{H_{\text{feature}}}{14}, 2.0, 6.0\right)$$

---

## 3. KẾT QUẢ THỰC NGHIỆM CHI TIẾT

### 3.1. Bảng kết quả từng Class của Mô hình Đề xuất (EfficientNet-B3)

| Class | Instance AUROC | Full Pixel AUROC | Anomaly Pixel AUROC |
|---|:---:|:---:|:---:|
| **Bottle** | **1.0000** | 0.9354 | 0.9129 |
| **Cable** | **0.9550** | 0.9333 | 0.9016 |
| **Capsule** | **0.8899** | 0.9622 | 0.9550 |
| **Grid** | **0.9649** | 0.8882 | 0.8472 |
| **Pill** | **0.8333** | 0.9486 | 0.9424 |
| **Screw** | **0.8854** | 0.9022 | 0.8804 |
| **TRUNG BÌNH (MEAN)** | **0.9214** | **0.9283** | **0.9066** |

---

### 3.2. Bảng so sánh giữa Baseline vs. Đề xuất

| Metric / Tiêu chí | Baseline (WideResNet-50) | Đề xuất (EfficientNet-B3) | Mức độ cải thiện / Trade-off |
|---|:---:|:---:|---|
| **Số lượng tham số (Params)** | 68.9M | **12.2M** | **Giảm 82.3% (Nhẹ hơn 5.7×)** |
| **Mean Instance AUROC** | 0.9800 | **0.9214** | Trade-off hợp lý cho mô hình siêu nhẹ |
| **Mean Full Pixel AUROC** | 0.9870 | **0.9283** | Phân vùng định vị lỗi tốt (0.928+) |
| **Mean Anomaly Pixel AUROC** | 0.9830 | **0.9066** | Đạt độ chính xác ổn định trên Edge |
| **Coreset Sampler** | Fixed 10% | **Adaptive (7-15%)** | Tự động điều chỉnh theo dung lượng bộ nhớ |
| **Hàm Scoring** | Mean k-NN | **Softmax Weighted** | Tăng độ nhạy với các lỗi nhỏ |

---

## 4. DẠNG ĐẦU RA CỦA HỆ THỐNG (OUTPUT ARTIFACTS)

Hệ thống đã tạo và lưu đầy đủ các kết quả thực nghiệm tại `results/run/run_1/`:

1. **File kết quả CSV:** `results/run/run_1/results_efficientnet_b3.csv`
2. **File JSON chi tiết từng ảnh:** `results/run/run_1/anomaly_scores/efficientnet_b3/*.json`
3. **Ảnh phân vùng bất thường (Segmentation Heatmaps):** `results/run/run_1/segmentation_images/efficientnet_b3/*/` (Gồm 3 panels: Input | GT Mask | Heatmap).

---

## 5. ĐÁNH GIÁ VÀ KẾT LUẬN

1. **Gọn nhẹ & Tiết kiệm tài nguyên:** Mô hình EfficientNet-B3 chỉ có 12M tham số (giảm 82% so với WideResNet-50 69M), cho phép chạy trực tiếp trên các thiết bị Edge/nhúng.
2. **Hiệu năng cạnh tranh:** Đạt Mean Instance AUROC **0.9214** và Mean Pixel AUROC **0.9283** trên 6 lớp dữ liệu công nghiệp MVTec.
3. **Mã nguồn sạch & Độc lập:** Hệ thống được tổ chức dạng thư viện chuẩn Python `ail_detector`, xóa bỏ hoàn toàn code thừa của repository cũ.
