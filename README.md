# AutoML Mini-Framework

[![CI](https://github.com/naufaldahafizh/automl-core/actions/workflows/ci.yml/badge.svg)](https://github.com/naufaldahafizh/automl-core/actions)

Framework modular untuk AutoML sederhana yang mampu:
- Melakukan preprocessing data
- Mencoba beberapa model machine learning
- Melakukan evaluasi otomatis (cross-validation)
- Menyimpan model terbaik & laporan evaluasi

---

## Studi Kasus

- **Dataset**: Breast Cancer (sklearn)
- **Tipe**: Binary Classification
- **Model**: Logistic Regression, Random Forest, XGBoost
- **Skoring**: Accuracy (default)
- **Evaluasi**: Cross-validation (CV=5)

---

## Struktur Proyek

```
automl-core/
├── src/
│ ├── preprocessor.py # Preprocessing & scaling
│ ├── model_selector.py # Model candidates
│ ├── evaluator.py # CV evaluator
│ └── automl_pipeline.py # Orkestrator utama
├── pipeline.py # CLI runner
├── tests/ # Unit tests
├── models/ # Model terbaik tersimpan
├── results/ # Laporan evaluasi JSON
├── requirements.txt
└── README.md
```

---

## Cara Menjalankan

### 1. Clone & Install

```bash
git clone https://github.com/naufaldahafizh/automl-core.git
cd automl-core
pip install -r requirements.txt
```

### 2. Jalankan pipeline
```bash
python pipeline.py
```

### 3. Jalankan unit tests
```bash
pytest tests/
```