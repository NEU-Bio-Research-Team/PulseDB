# Tổng hợp artifacts & trạng thái pipeline (2026-02-08)

Tài liệu này tổng hợp những gì rút ra được từ toàn bộ kết quả trong `pulsedb_subset/artifacts/` sau chuỗi thay đổi vừa thực hiện (fix schema/IO, QC, khóa PPG-only, và subject-disjoint split). Mục tiêu: tạo “Data/Preprocessing + Protocol” đủ chặt để đưa vào paper.

---

## 1) TL;DR (đã đạt được gì)

- **Schema/manifest đã “đúng và đủ”**: `subset_manifest.csv` hiện có đầy đủ đường dẫn HDF5 `/Subset/*` (Signals/SBP/DBP/Subject/Age/...) và các kích thước chuẩn (ví dụ Train: `1250 x 3 x 465480`).
- **PPG-only được khóa cứng**: config yêu cầu `channels=[PPG]` và `channel_indices.PPG=1` (theo debug channels: ch0=ECG, ch1=PPG, ch2=ABP). QC metrics ghi rõ `channel_name=PPG`, `channel=1`.
- **Waveform QC đã có baseline**: `qc_metrics.csv` + report `eda_waveform_qc.md` cung cấp thống kê QC (nan/flatline/saturation + `snr_variance_proxy`) trên mẫu 20k/ subset (AAMI_Test dùng toàn bộ 666 segment).
- **Leakage theo subject đã được “đưa vào protocol” bằng split toàn cục**:
  - `subject_splits.yaml`: danh sách subject cho train/val/test theo seed.
  - `split_stats_by_subset.csv`: số segment + số subject của từng subset nằm trong từng split.
  - `subject_overlap.csv`: overlap subject giữa các subset (đáng chú ý Train và CalBased_Test **trùng subject hoàn toàn**, nên không thể coi CalBased_Test là test độc lập nếu không filter theo split).

**Ý nghĩa:** Pipeline hiện đã ở trạng thái **Data curation + Protocol locking** (chuẩn bị dữ liệu/EDA/QC/splitting) — đủ để viết phần *Dataset / Preprocessing / Quality Control / Data leakage prevention*. Bước kế tiếp là **chạy preprocess/cache theo split** và bắt đầu **training/evaluation**.

---

## 2) Inventory & data dictionary (từ `subset_manifest.csv`)

Nguồn chân lý về cấu trúc dữ liệu:
- Mỗi subset là MAT v7.3 (HDF5)
- `Signals` có dạng **`(T=1250, C=3, N)`** ở thô (trong file)
- Nhãn `SBP`, `DBP`, `Subject`, và demographics có kích thước theo segment (N)

Ví dụ (trích từ `subset_manifest.csv`):
- Train: `Signals` = `1250 x 3 x 465480`, `SBP/DBP/Subject` = `N=465480`
- CalBased_Test: `N=51720`
- CalFree_Test: `N=57600`
- AAMI_Cal: `N=70212`
- AAMI_Test: `N=666`

**Ý nghĩa trong paper:** Đây là “bảng mô tả dataset” (data dictionary) để chứng minh:
- File có đủ trường cần thiết
- Kích thước đồng nhất theo segment
- Segment length = 10s (fs=125Hz → 1250 samples)

---

## 3) Labels + demographics EDA (từ `stats_by_subset.csv`, `missingness_by_subset.csv`)

### 3.1. Phân phối nhãn SBP/DBP
Tóm tắt thống kê (mean/std/quantiles) đã có trong `stats_by_subset.csv`:
- Train/CalBased/CalFree có thống kê SBP/DBP khá tương đồng.
- **AAMI_Test có SBP/DBP cao hơn rõ** (mean SBP ~134.7, DBP ~75.4) → gợi ý **distribution shift** (có thể cần thảo luận trong paper và/hoặc calibration strategy).

### 3.2. Missingness demographics
`missingness_by_subset.csv` cho thấy các trường `age/gender/height/weight/bmi` có **missing_rate = 0.0** cho tất cả subset.

**Ý nghĩa trong paper:**
- Có thể mô tả demographics scaler rõ ràng (không phải xử lý missing phức tạp).
- Nhưng vẫn nên nêu rõ: demographics chỉ là covariates; mục tiêu chính vẫn PPG→BP.

> Lưu ý: report `reports/eda_labels_demographics.md` có thể đã được tạo trước khi fix decoder subject_id (nó báo overlap nhỏ). Nếu cần đưa số overlap vào paper, nên **rerun script 02** để đồng bộ với decoder mới.

---

## 4) Waveform QC (từ `qc_metrics.csv`, `eda_waveform_qc.md`)

### 4.1. QC metrics đang có
`qc_metrics.csv` chứa các trường (per segment):
- `nan_ratio`: tỉ lệ NaN trong waveform
- `flatline_ratio`: tỉ lệ điểm “phẳng” (ít biến thiên)
- `saturation_ratio`: tỉ lệ điểm bão hòa/clipping
- `snr_variance_proxy`: proxy SNR dựa trên tỉ lệ variance (không phải SNR vật lý)

Report `eda_waveform_qc.md` tóm tắt theo subset (mẫu 20k/ subset):
- `nan_ratio_mean ≈ 0`
- `flatline_mean ≈ 0.002`
- `saturation_mean ≈ 0.0016`
- `snr_p50 ≈ 34–35` (khá ổn và tương đồng giữa các subset)

### 4.2. PPG-only đã được thể hiện trong artifact
Trong `qc_metrics.csv` đã ghi rõ:
- `channel=1`, `channel_name=PPG`

**Ý nghĩa trong paper:** Đây là phần *Signal Quality Control*:
- Nêu rõ metrics + ngưỡng QC (từ config)
- Nêu rõ sampling strategy (20k/ subset) và ngoại lệ (AAMI_Test chỉ 666)

---

## 5) Subject leakage & protocol split (cụm artifacts quan trọng nhất)

### 5.1. Overlap giữa subsets (từ `subject_overlap.csv`)
Điểm chính:
- **Train ↔ CalBased_Test: overlap = 1293/1293 (trùng subject hoàn toàn)**
- Train ↔ (CalFree_Test, AAMI_Test, AAMI_Cal): overlap = 0
- **AAMI_Test ↔ AAMI_Cal: overlap = 116/116 (trùng subject hoàn toàn)**

**Ý nghĩa:**
- Không thể dùng “tên subset” như một bảo chứng cho generalization.
- Muốn claim *subject generalization*, bắt buộc phải filter theo split subject-level.

### 5.2. Subject-disjoint split toàn cục (từ `subject_splits.yaml`)
- Tổng subjects: 1553
- Train/val/test theo seed=42 và fractions 0.8/0.1/0.1

### 5.3. Thống kê split theo subset (từ `split_stats_by_subset.csv`)
Ví dụ:
- Train subset: train/val/test segments = 375,840 / 46,080 / 43,560; subjects = 1044 / 128 / 121
- AAMI_Test subset: train/val/test segments = 515 / 73 / 78; subjects = 90 / 12 / 14

**Ý nghĩa trong paper (quan trọng):**
- Đây là “bảng split” có thể đưa vào phụ lục (appendix) hoặc methods.
- Khẳng định pipeline đánh giá theo split (không theo subset name).

---

## 6) Đang ở stage nào trong paper?

Hiện tại pipeline tương ứng với các phần sau trong paper:

1) **Dataset & Signal description**
- Segment length, sampling rate, tensor convention
- Kênh PPG-only (định danh kênh + enforcement)

2) **Preprocessing & Quality Control**
- QC metrics + thresholds (đã có artifacts)
- Log “dropped segments” (sẽ đầy đủ sau khi chạy preprocess/cache)

3) **Evaluation protocol (anti-leakage)**
- Subject-disjoint split toàn cục + thống kê split
- Nêu rõ rủi ro: Train và CalBased_Test dùng cùng subjects; AAMI_Cal và AAMI_Test dùng cùng subjects

**Chưa bước vào phần “Modeling/Results”** (training, metrics MAE/RMSE, AAMI criteria, ablation, calibration strategy, v.v.).

---

## 7) Hướng tiếp theo làm gì? (đề xuất checklist)

### 7.1. Hoàn tất dữ liệu cho training (bắt buộc)
- Chạy preprocess/cache theo split:
  - `python pulsedb_subset/scripts/04_preprocess_cache_shards.py`
- Kiểm tra outputs:
  - `artifacts/scalers.json`
  - `artifacts/dropped_segments.csv` (drop rate theo QC + label range)
  - Cache shards theo thư mục: `cache/pulsedb_subset/{train|val|test}/{SubsetName}/...`

### 7.2. Đồng bộ lại EDA report (khuyến nghị)
- Rerun labels/demographics EDA để số liệu overlap trong report khớp với decoder mới:
  - `python pulsedb_subset/scripts/02_eda_labels_demographics.py`

### 7.3. Bắt đầu modeling (giai đoạn paper “Experiments/Results”)
Tối thiểu cần:
- Một training script đọc shards `train`, validate trên `val`, test trên `test`.
- Metric: MAE/RMSE cho SBP/DBP (và có thể theo bin SBP như config `sbp_bins`).
- Protocol viết rõ: *fit scaler chỉ trên Train∩train-split*, không dùng test.

### 7.4. Quy ước báo cáo cho paper
- Table 1: thống kê dataset (N segments, N subjects) theo split/subset.
- Table QC: drop rates và lý do drop.
- Figure: ví dụ waveform QC best/worst (PPG-only).

---

## 8) Danh sách artifacts chính để trích dẫn/đính kèm

- Schema/manifest:
  - `artifacts/schema_map.json`
  - `artifacts/subset_manifest.csv`
- Labels + demo:
  - `artifacts/stats_by_subset.csv`
  - `artifacts/missingness_by_subset.csv`
  - `artifacts/plots/labels/*.png`
- Waveform QC:
  - `artifacts/qc_metrics.csv`
  - `reports/eda_waveform_qc.md`
  - `artifacts/plots/waveform_qc/*.png`
- Anti-leakage protocol:
  - `artifacts/subject_splits.yaml`
  - `artifacts/split_stats_by_subset.csv`
  - `artifacts/subject_overlap.csv`

---

### Ghi chú “paper-grade” (ngắn)
- Khi viết claim generalization: luôn nói rõ **subject-disjoint split** (không dựa vào subset name).
- AAMI_Cal và AAMI_Test dùng cùng subjects: phù hợp cho calibration-on-subject nhưng phải mô tả đúng (không nhầm thành subject-generalization).
