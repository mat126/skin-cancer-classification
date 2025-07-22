# 📁 Data

This folder is intended to store the **input data** used in the project  
🧠 *Skin Cancer Classification using Deep Learning (ISIC 2024)*.

---

## ⚠️ Important Notice

Due to file size and licensing restrictions, **raw image data and metadata are not included** in this repository.

If you wish to reproduce or explore this project, you must **manually download the data** from the official Kaggle competition page:

🔗 **ISIC 2024 Challenge – Skin Lesion Diagnosis**  
https://www.kaggle.com/competitions/isic-2024-challenge

---

## 📦 Expected Folder Structure (once downloaded)

After downloading and unzipping the data from Kaggle, place the files as follows:

```bash
data/
├── train/
│   ├── <image_1>.jpg
│   ├── <image_2>.jpg
│   └── ...
├── test/
│   ├── <image_1>.jpg
│   └── ...
├── train-metadata.csv
├── test-metadata.csv
└── sample_submission.csv

## 📄 Variable Dictionary

A detailed description of all variables in the metadata files (e.g., `train-metadata.csv`) is available here:  
➡️ [data_description.md](./data_description.md)

