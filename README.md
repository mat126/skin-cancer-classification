# 🧠 Skin Cancer Classification (ISIC 2024)

This project explores the classification of skin lesions using deep learning and machine learning techniques.  
It aims to fuse **dermoscopic image data** with **structured clinical metadata** to improve melanoma detection and prediction accuracy.

---

## 📦 Project Status

![Status](https://img.shields.io/badge/Project-WIP-orange)  
🚧 **Work in progress** – Hybrid CNN + metadata model under development.  
Classical models (Random Forest, XGBoost) and full EDA completed.

---

## 📁 Dataset

The dataset comes from the [Kaggle ISIC 2024 Challenge](https://www.kaggle.com/competitions/isic-2024-challenge).  
Due to licensing and file size constraints, **raw image data is not included**.

📄 See [data/README.md](data/README.md) for download instructions and expected folder structure.

---

## 🎯 Objectives

- [x] Perform exploratory data analysis (EDA) on image metadata
- [x] Build preprocessing pipelines for clinical variables
- [x] Train baseline models: Random Forest and XGBoost
- [ ] Develop a hybrid CNN for multimodal classification
- [ ] Apply model explainability (e.g., SHAP) to both image and metadata features
- [ ] Compare image-only, metadata-only, and hybrid models

---

## 🧰 Tools & Libraries

- **Languages**: Python  
- **Frameworks**: TensorFlow / Keras, Scikit-learn, XGBoost  
- **Visualization**: Matplotlib, Seaborn  
- **Utilities**: NumPy, Pandas, OpenCV, UMAP, t-SNE  

---

## 📁 Project Structure

```
skin-cancer-classification/
├── data/                 # Raw and processed data (not included in repo)
├── notebooks/            # Jupyter notebook(s) for training
├── src/                  # Source code: preprocessing, models, EDA utils
├── models/               # Trained model files (Pickle)
├── references/           # BibTeX files and research sources
├── CV_Matte_Morella.pdf  # Author's CV
├── LICENSE
├── README.md
└── .gitignore
```

---

## ✍️ Author

**Matteo Morella**  
📧 Email: mmorella9@gmail.com  
🐙 GitHub: [@mat126](https://github.com/mat126)  
📄 CV: [Download CV - PDF](https://raw.githubusercontent.com/mat126/mat126/main/Cv_Matteo_Morella.pdf)
