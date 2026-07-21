# Data Balancing

Folder này tập trung vào 2 việc:

1. **Train + đánh giá imbalance methods**: chạy các cách `original`,
   `undersampling`, `smote`, `class_weights` với nhiều model rồi so sánh kết quả.
2. **Monitor dữ liệu đến**: chọn checkpoint đã train, score transaction mới,
   đo latency, xem class dự đoán, xem chi tiết từng điểm dữ liệu và kiểm tra drift.

`balancing_experiments.py` là nơi train/evaluate. `monitor.py` và
`monitor_app.py` chỉ dùng để monitor, không train lại model.

## 1. Setup

Từ repo root:

```bash
conda activate BA
pip install -r data_balancing/requirements.txt
```

Các lệnh phía dưới mặc định đã activate env `BA`.

## 2. Yêu Cầu Input

File cần có trước khi chạy:

```text
feature_engineering/fraud-detection/data/processed/transactions_cleaned.parquet
```

Kiểm tra nhanh data hiện tại:

```bash
python data_balancing/data_source_check.py
```

Kết quả mong muốn:

```text
Active raw: full_kaggle_candidate
Cleaned data: full_like
Cleaned rows: 6,362,620
Fraud rows: 8,213
Fraud rate: ~0.1291%
```

Nếu thiếu raw data, tải đúng Kaggle dataset:

```bash
mkdir -p feature_engineering/fraud-detection/data/raw
kaggle datasets download -d rupakroy/online-payments-fraud-detection-dataset -p feature_engineering/fraud-detection/data/raw --unzip
ln -sf PS_20174392719_1491204439457_log.csv feature_engineering/fraud-detection/data/raw/online_fraud_detection.csv
```

Nếu cần build lại cleaned data:

```bash
cd feature_engineering/fraud-detection
PYTHONPATH=src python src/generate_synthetic.py
PYTHONPATH=src python src/cleaning.py
cd ../..
```

## 3. Chạy Train Và Đánh Giá

Chạy nhanh để smoke test:

```bash
python data_balancing/balancing_experiments.py --max-rows 10000 --fraud-review-size 200
```

Chạy full data:

```bash
python data_balancing/balancing_experiments.py --max-rows 0 --split stratified --run-cv --cv-folds 5
```

Chạy full data theo time split:

```bash
python data_balancing/balancing_experiments.py --max-rows 0 --split time --run-cv --cv-folds 5
```

Script sẽ so sánh:

| Method | Ý nghĩa |
| --- | --- |
| `original` | Data ban đầu, không balancing. |
| `undersampling` | Giảm bớt class non-fraud trong train set. |
| `smote` | Tạo fraud synthetic rows trong train set. |
| `class_weights` | Giữ nguyên data, tăng penalty cho fraud class. |

Lưu ý: SMOTE chỉ được fit/resample một lần trên train set rồi reuse cho các model
trong strategy `smote`, không tạo lại data cho từng model.

Models được chạy:

- Logistic Regression
- Random Forest
- XGBoost
- HistGradientBoostingClassifier

Metric chính:

- `auc_pr`
- `fraud_cases_captured`
- `fraud_cases_missed`
- `fraud_case_capture_rate`
- `fraud_amount_captured`
- `fraud_amount_missed`
- `fraud_value_capture_rate`
- `review_queue_size`
- `false_alerts`
- `business_cost`
- `operating_threshold`

Precision/recall/F1 vẫn được lưu để tham chiếu, nhưng khi test cần ưu tiên câu
hỏi: model phát hiện được bao nhiêu case fraud và bỏ sót bao nhiêu case fraud.
Accuracy không dùng làm metric chính vì fraud quá ít.

## 4. Outputs Sau Khi Train

Tất cả output nằm trong:

```text
data_balancing/outputs/
```

File quan trọng:

| File | Nội dung |
| --- | --- |
| `balancing_model_results.csv` | Bảng so sánh method/model. |
| `balancing_report.md` | Report tổng hợp. |
| `fraud_review_results.csv` | Review set tập trung vào fraud rows. |
| `feature_target_scores.csv` | Mutual information của feature với `isFraud`. |
| `model_permutation_importance.csv` | Feature importance của best model. |
| `run_metadata.json` | Cấu hình run và feature columns. |
| `artifacts/best_balancing_model.joblib` | Best model checkpoint. |
| `artifacts/model_checkpoints.csv` | Manifest các checkpoint đã train. |
| `artifacts/checkpoints/*.joblib` | Checkpoint cho từng cặp method/model. |

Đọc nhanh kết quả:

```text
data_balancing/outputs/balancing_report.md
data_balancing/outputs/balancing_model_results.csv
```

## 5. Monitor Dữ Liệu Đến

Phần monitor có 2 cách dùng trong cùng một luồng:

- Chạy `monitor.py` để tạo reference/drift report dạng file.
- Chạy `monitor_app.py` để xem dashboard realtime, chọn checkpoint và click từng transaction.

Build reference distribution:

```bash
python data_balancing/monitor.py --build-reference --max-reference-rows 100000
```

Check drift cho current window:

```bash
python data_balancing/monitor.py --max-current-rows 50000
```

Monitoring outputs:

| File | Nội dung |
| --- | --- |
| `monitoring_reference.parquet` | Reference distribution. |
| `monitoring_drift_report.csv` | Drift report theo feature. |
| `monitoring_summary.md` | Tóm tắt drift. |
| `figures/monitoring_drift_summary.png` | Hình top drift signals. |

Drift metric:

- Numeric feature: PSI và KS statistic.
- Categorical feature: distribution shift.
- `warning`: bắt đầu drift.
- `alert`: drift mạnh, cần kiểm tra.

Mở dashboard:

```bash
streamlit run data_balancing/monitor_app.py
```

Lưu ý Linux dùng `/`, không dùng `\`.

Các tab chính:

- `Home`: chọn scenario, bấm `Start`, stream tự chạy liên tục; click/chọn từng `tx_id` để xem detail.
- `Feature Drift`: xem kết luận tổng quan, bảng feature cần chú ý trước, rồi mở từng feature để xem reference/current khác nhau ở đâu.
- `Models`: so sánh kết quả train/evaluate.
- `Importance`: feature importance.
- `Config`: data source check, checkpoint, threshold, drift config.
- `Guide`: hướng dẫn và giải thích các realtime synthesis modes.

Realtime synthesis modes:

| Mode | Ý nghĩa |
| --- | --- |
| `Normal traffic` | Lấy mẫu từ reference, không inject bất thường. |
| `Account takeover` | New device, mismatch, failed attempts, IP distance tăng. |
| `High-value cashout` | Amount tăng, type chuyển về `CASH_OUT`. |
| `Bot burst` | Time gap thấp, transaction count và failed attempts tăng. |
| `Foreign IP wave` | IP mismatch và billing distance tăng. |
| `Mixed attack` | Trộn nhiều pattern. |

Trong tab `Home`, các control quan trọng:

- `Scenario preset`: cấu hình nhanh cho người mới, ví dụ demo dễ quan sát hoặc traffic bình thường.
- `Transactions per tick`: số transaction mới được thêm mỗi giây.
- `Transactions/min`: nhịp timestamp mô phỏng của transaction.
- `Target fraud rate`: tỉ lệ fraud thật mong muốn trong stream realtime.
- `Attack share`: tỉ lệ transaction được inject pattern bất thường để test drift/scoring.
- `Visible buffer`: số điểm gần nhất giữ lại trên chart/table để không bị quá dày.
- `Start`: bắt đầu tự sinh dữ liệu liên tục.
- `Pause`: dừng stream nhưng vẫn giữ dữ liệu hiện tại để xem chi tiết.
- `Reset`: xóa buffer hiện tại và reset lại stream.

Trong khu vực transaction realtime:

- `Khung thời gian`: xem `5 phút`, `30 phút`, `1 giờ`, `1 ngày`, `1 tuần`,
  `1 tháng`, `1 năm` hoặc `Từ lúc bắt đầu`.
- Có thể filter theo mã transaction, account/name, IP, type, class dự đoán,
  fraud thật và ngày đến.
- Click trực tiếp vào điểm trên chart hoặc click row trong bảng để chọn
  transaction; phần detail bên dưới sẽ đổi theo transaction đang chọn.
- Bảng transaction có pagination, tối đa 50 row/page. Mặc định sort `Cũ -> mới`
  để trang đang xem ít bị nhảy khi stream append dữ liệu mới.

Trong tab `Feature Drift`, đọc theo thứ tự:

1. Xem `Kết luận`, `Lệch mạnh`, `Cần theo dõi`, `Ổn định`.
2. Xem bảng `Feature cần chú ý trước`; ưu tiên các dòng `Lệch mạnh`.
3. Mở từng feature để xem thống kê reference/current và biểu đồ phân phối.
4. Drift không tự kết luận fraud, nhưng báo dữ liệu realtime đã khác data ban đầu.
5. Biểu đồ top drift dùng `Mức lệch hiển thị` từ 0 đến 1 để dễ nhìn; điểm raw
   vẫn nằm trong bảng.

Trong tab `Models`:

- Nếu test chỉ có vài fraud case, dashboard sẽ cảnh báo đây là smoke/sample run,
  không nên kết luận model.
- `Holdout test` dùng để so sánh model trên dữ liệu chưa train.
- `K-fold validation` dùng để xem model có ổn định giữa các fold không, không
  phải test set.
- Có nút `Retrain` để chạy lại `balancing_experiments.py` nền từ dashboard; nên
  chọn `Full data`, `stratified`, bật `Run k-fold validation`, và để `CV folds=5`.

## 6. Lệnh Nên Chạy Theo Thứ Tự

```bash
python data_balancing/data_source_check.py
python data_balancing/balancing_experiments.py --max-rows 0 --split stratified --run-cv --cv-folds 5
python data_balancing/monitor.py --build-reference --max-reference-rows 100000
python data_balancing/monitor.py --max-current-rows 50000
streamlit run data_balancing/monitor_app.py
```
