# AIL Anomaly Detector

Phát hiện và định vị bất thường (anomaly detection & segmentation) trong ảnh công nghiệp sử dụng memory-bank patch embedding với các cải tiến của nhóm AIL.

---

## Kiến trúc tổng quan

```
Input Image
    ↓
Pretrained Backbone (EfficientNet-B3 / WideResNet-50 / DINOv2)
    ↓
Multi-layer Feature Extraction (forward hooks)
    ↓
Patch Embedding + Preprocessing (MeanMapper → Aggregator)
    ↓
AdaptiveCoreset Subsampling ← cải tiến: tự động chọn tỷ lệ nén
    ↓  (memory bank built during training)
SoftmaxNNScorer ← cải tiến: softmax-weighted k-NN distance
    ↓
AdaptiveRescaleSegmentor ← cải tiến: sigma Gaussian adaptive
    ↓
Anomaly Score (image-level) + Segmentation Map (pixel-level)
```

## Các cải tiến so với phương pháp gốc

| Thành phần | Phương pháp gốc | Cải tiến của nhóm |
|---|---|---|
| Backbone mặc định | WideResNet-50 (69M params) | **EfficientNet-B3 (12M params)** |
| Coreset sampling | Fixed 10% | **AdaptiveCoreset** – tự chọn 7/10/15% theo dataset size |
| NN Scoring | Mean k-NN distance | **SoftmaxNNScorer** – softmax-weighted distance |
| Segmentation | Fixed Gaussian σ=4 | **Adaptive σ** theo feature map resolution |
| Output | AUROC metrics | + **per-image JSON scores** + heatmap PNG |

---

## Cài đặt

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows

pip install -e .
```

---

## Sử dụng

### Chạy với EfficientNet-B3 (nhẹ, mặc định)

```bash
python train_local.py
```

### Chạy với WideResNet-50

```bash
python train_local.py --backbone wideresnet50
```

### So sánh hai backbone

```bash
python train_local.py --backbone efficientnet_b3,wideresnet50 --compare
```

### Lưu kết quả đầy đủ (score JSON + segmentation PNG)

```bash
python train_local.py --save_images --output_scores
```

### Chỉ test một class

```bash
python train_local.py --classes bottle --save_images --output_scores
```

---

## Output

Mỗi lần chạy tạo ra thư mục `results/run/run_N/` chứa:

```
run_N/
├── efficientnet_b3/
│   └── results.csv             ← AUROC per class
├── results_efficientnet_b3.csv ← full results
├── results_comparison.csv      ← so sánh backbones (nếu nhiều)
├── anomaly_scores/             ← (nếu --output_scores)
│   └── efficientnet_b3/
│       └── bottle_scores.json  ← per-image score
└── segmentation_images/        ← (nếu --save_images)
    └── efficientnet_b3/
        └── bottle/
            └── *.png
```

### Format `anomaly_scores.json`

```json
{
  "class": "bottle",
  "backbone": "efficientnet_b3",
  "records": [
    {
      "image_path": "data/bottle/test/broken_large/000.png",
      "anomaly_score": 0.823456,
      "is_anomaly_gt": 1,
      "is_anomaly_pred": 1,
      "threshold": 0.5
    },
    ...
  ]
}
```

---

## Tùy chọn dòng lệnh

| Tùy chọn | Mặc định | Mô tả |
|---|---|---|
| `--backbone` | `efficientnet_b3` | Tên backbone (có thể dùng nhiều, phân cách bởi dấu phẩy) |
| `--classes` | *(tất cả)* | Tên class cần train |
| `--compare` | false | In bảng so sánh AUROC khi dùng nhiều backbone |
| `--layers` | *(tự động)* | Override tên layer cần hook |
| `--coreset_mode` | `auto` | `auto` = AdaptiveCoreset; hoặc số thực như `0.1` |
| `--target_dim` | 1024 | Chiều embedding memory bank |
| `--patchsize` | 3 | Kích thước patch kernel |
| `--num_nn` | 5 | Số neighbour cho scoring |
| `--temperature` | 1.0 | Nhiệt độ softmax scorer |
| `--batch_size` | 2 | Batch size DataLoader |
| `--imagesize` | 224 | Kích thước ảnh sau crop |
| `--resize` | 256 | Kích thước resize |
| `--gpu` | 0 | GPU index (-1 = CPU) |
| `--save_models` | false | Lưu model |
| `--save_images` | false | Lưu ảnh segmentation |
| `--output_scores` | false | Lưu per-image scores JSON |

---

## Kết quả thực nghiệm

Chạy trên 6 class MVTec-AD: `bottle`, `cable`, `capsule`, `grid`, `pill`, `screw`

| Backbone | Instance AUROC | Pixel AUROC | Params | Ghi chú |
|---|---|---|---|---|
| WideResNet-50 | 0.9800 | 0.9870 | 69M | Baseline paper |
| EfficientNet-B3 | ~0.972 | ~0.982 | 12M | **Nhẹ hơn 5×** |

> EfficientNet-B3 đạt pixel AUROC cạnh tranh nhờ AdaptiveRescaleSegmentor trong khi giảm đáng kể chi phí tính toán.

---

## Backbone mặc định

| Backbone | Layers hook | Params | Ghi chú |
|---|---|---|---|
| `efficientnet_b3` | `blocks.5, blocks.6` | 12M | **Mặc định – nhẹ, nhanh** |
| `wideresnet50` | `layer2, layer3` | 69M | Baseline so sánh |
| `dinov2_vitb14` | `blocks.9, blocks.11` | 86M | Chất lượng cao nhất |
