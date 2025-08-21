# 🧠 Skin Cancer Classification (ISIC 2024)

This project explores the classification of skin lesions using deep learning and machine learning techniques.  
It fuses **dermoscopic image data** with **structured clinical metadata** to improve melanoma detection.

---

## 📦 Project Status

![Status](https://img.shields.io/badge/Project-WIP-orange)
![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)

- ✅ EDA on image metadata  
- ✅ Preprocessing pipelines for clinical variables  
- ✅ Baseline models: Random Forest, XGBoost  
- ✅ K-fold stacking (RF, XGB, FFN) + logistic meta-model  
- ✅ Initial CNN trial (pretrained backbone)  
- ⏳ Hybrid CNN (images + metadata)  
- ⏳ Explainability (e.g., SHAP)  
- ⏳ Full comparison: image-only vs metadata-only vs hybrid

---

## 📁 Dataset

The dataset comes from the **[Kaggle ISIC 2024 Challenge](https://www.kaggle.com/competitions/isic-2024-challenge)**.  
Due to licensing and size constraints, **raw images are not included**.

📄 See **[data/README.md](data/README.md)** for download instructions and the expected folder structure.

---

## 🚀 Reproduce

```bash
# 1) Environment
pip install -r requirements.txt

# 2) Data
#   - Follow data/README.md to place CSVs and images under data/

# 3) Stacking (RF + XGB + FFN → Logistic meta)
python -m src.train.stacking_kfold

# 4) CNN trial (pretrained backbone)
python -m src.train.train_cnn
```

**Outputs & Models**
- Small artifacts (metrics JSON/CSV, a few PNG figures) will go under `outputs/`.
- Trained models (`.pkl`, `.keras`) are saved under `models/`.  
  Large files are excluded from Git; consider GitHub Releases or external storage and link them here.

---

## 🧰 Tools & Libraries

- **Languages**: Python  
- **Frameworks**: TensorFlow/Keras, Scikit-learn, XGBoost  
- **Visualization**: Matplotlib, Seaborn  
- **Utilities**: NumPy, Pandas, UMAP, t-SNE, OpenCV (optional)

---

## 📂 Project Structure

```
skin-cancer-classification/
├── data/                 # Raw data (not in repo) + README with instructions
├── notebooks/            # Notebooks as reports (call into src/)
├── src/                  # Source code (pipelines, models, training, eval)
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── eda_utils.py
│   ├── models/
│   │   ├── cnn.py        # CNN / hybrid CNN builders
│   │   └── tabular.py    # RF, XGB, FFN builders
│   ├── train/
│   │   ├── train_cnn.py
│   │   └── stacking_kfold.py
│   ├── eval/
│   │   ├── metrics.py
│   │   └── plots.py
│   └── utils/
│       ├── seeding.py
│       └── io.py
├── models/               # Trained models (ignored by git; keep README)
├── outputs/              # Curated metrics/figures (small files)
├── references/           # bibliography.md and related notes
├── LICENSE
├── README.md
└── .gitignore
```

---

## 🔎 Notes on Reproducibility

- Seeds fixed where applicable (`src/utils/seeding.py`).  
- Configurable parameters in `src/config.py`. 
- If you use or adapt external utilities, keep clear attribution and license notes.

---

## 🙌 Credits

Some utilities and ideas are inspired by **Ilya Novoselskiy** — <https://github.com/ilyanovo>.  
See **CREDITS.md** and respect original licenses when reusing third-party code.

---

## ✍️ Author

**Matteo Morella**  
📧 Email: mmorella9@gmail.com  
🐙 GitHub: [@mat126](https://github.com/mat126)  
📄 CV: [Download CV – PDF](https://raw.githubusercontent.com/mat126/mat126/main/Cv_Matteo_Morella.pdf)
