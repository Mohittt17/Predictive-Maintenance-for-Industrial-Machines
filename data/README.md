# Dataset Download Instructions

This directory stores all raw datasets.  **Do not commit raw data to Git.**
Add `data/raw/` to `.gitignore`.

---

## 1. NASA C-MAPSS (Turbofan Engine Degradation Simulation)

**Used for**: RUL regression baseline, failure prediction baselines (M1–M6)

### Download
1. Visit: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
2. Find **"Turbofan Engine Degradation Simulation Data Set"**
3. Or direct mirror (Kaggle):
   ```
   kaggle datasets download -d behrad3d/nasa-cmaps
   ```
4. Extract into `data/raw/cmapss/` so the directory looks like:
   ```
   data/raw/cmapss/
   ├── train_FD001.txt
   ├── train_FD002.txt
   ├── train_FD003.txt
   ├── train_FD004.txt
   ├── test_FD001.txt
   ├── test_FD002.txt
   ├── test_FD003.txt
   ├── test_FD004.txt
   ├── RUL_FD001.txt
   ├── RUL_FD002.txt
   ├── RUL_FD003.txt
   └── RUL_FD004.txt
   ```

### Sub-dataset summary
| ID | Train engines | Test engines | Conditions | Fault modes |
|----|---------------|--------------|------------|-------------|
| FD001 | 100 | 100 | 1 | 1 |
| FD002 | 260 | 259 | 6 | 1 |
| FD003 | 100 | 100 | 1 | 2 |
| FD004 | 248 | 249 | 6 | 2 |

---

## 2. NASA IMS Bearing Dataset

**Used for**: Bearing anomaly detection, bearing RUL (M2–M7)

### Download
1. Visit: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
2. Find **"IMS Bearing Data Set"** (also called "University of Cincinnati IMS Bearing Dataset")
3. Or via Kaggle:
   ```
   kaggle datasets download -d vinayak123tyagi/bearing-dataset
   ```
4. Extract into `data/raw/ims/` so the directory looks like:
   ```
   data/raw/ims/
   ├── Set1/
   │   ├── 2003.10.22.12.06.24
   │   ├── 2003.10.22.12.09.13
   │   └── ... (2,156 files)
   ├── Set2/
   │   └── ... (984 files)
   └── Set3/
       └── ... (6,324 files)
   ```

### Run summary
| Run | Duration | Failure bearing | Failure mode |
|-----|----------|-----------------|--------------|
| Set1 | ~35 days | B3 outer race, B4 rolling element | Multiple |
| Set2 | ~6 days  | B1 outer race | Single |
| Set3 | ~45 days | B3 outer race | Single |

---

## 3. XJTU-SY Bearing Dataset

**Used for**: Richer bearing RUL, multimodal feature comparison (M5–M7)

### Download
1. Visit: https://biaowang.tech/xjtu-sy-bearing-datasets/
2. Or request via form at Xi'an Jiaotong University
3. Kaggle mirror:
   ```
   kaggle datasets download -d uysimty/bearing-dataset-xjtu-sy
   ```
4. Extract into `data/raw/xjtu/` so the directory looks like:
   ```
   data/raw/xjtu/
   ├── Condition_1/
   │   ├── Bearing1_1/
   │   │   ├── 1.csv
   │   │   ├── 2.csv
   │   │   └── ...
   │   ├── Bearing1_2/
   │   └── ...
   ├── Condition_2/
   └── Condition_3/
   ```

### Condition summary
| Condition | Speed (RPM) | Radial load (kN) | Bearings |
|-----------|-------------|------------------|----------|
| Condition_1 | 2100 | 12 | 5 |
| Condition_2 | 2250 | 11 | 5 |
| Condition_3 | 2400 | 10 | 5 |

---

## Verify Download

After downloading, run from the project root:

```bash
python -c "
from src.ingestion.cmapss_loader import load_cmapss
train, test = load_cmapss('FD001')
print(f'C-MAPSS FD001: {train.n_readings} train rows, {test.n_readings} test rows')
"
```

```bash
python -c "
from src.ingestion.ims_loader import load_ims
ds = load_ims('Set1', bearing_channel=1)
print(f'IMS Set1 B1: {ds.n_readings} snapshots')
"
```

```bash
python -c "
from src.ingestion.xjtu_loader import load_xjtu
ds = load_xjtu('Condition_1', 'Bearing1_1')
print(f'XJTU C1 B1_1: {ds.n_readings} minute files')
"
```
