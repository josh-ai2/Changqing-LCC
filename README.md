## LSTM-Based Tree Species Classification for Changqing Region

### Abstract

This project demonstrates an **LSTM-based approach** for classifying tree species in the Changqing region using **remote sensing time-series data**. The workflow involves extracting multi-month Sentinel-2 features from Google Earth Engine (GEE), cleaning and structuring these data locally, and then training a **Long Short-Term Memory (LSTM)** neural network to classify each pixel’s dominant tree species. The final output is a **map** highlighting species distributions across the study site. This approach uses a **spatiotemporal** approach to classify forest and tree species types focused on biodiversity analysis. We mainly classify 

---

## Overview of the Workflow (main-workflow-latest.ipynb)

1. **Time-Series Data Extraction (GEE)**
   - We utilized **Google Earth Engine (GEE)** to extract monthly Sentinel-2 spectral bands and indices (e.g. NDVI, NDWI, EVI) for each pixel over a 12-month period.
   - Datases includes:  
     - **Spectral Bands**: B1, B2, B3, ..., B12  
     - **Indices**: NDVI, NDWI, EVI  
     - **Other**: AOT, TCI, WVP, etc.
     - **Earth Engine Snippet: ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")** [Link]([/guides/content/editing-an-existing-page](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED))

2. **Local Data Cleaning & Organization**
   - The raw CSV files from GEE were **cleaned** locally to remove incomplete or erroneous records. 
   - The final structure includes 12 time steps (one row per month) for each pixel `(lat, lon)`, plus the “Class” label in groundtruth data.

3. **LSTM Model for Classification**
   - We built a **PyTorch** LSTM model (`LSTMClassifier`) to handle the 12×(20+) time-series features. We train on the groundtruth data, then perform classification on our GEE unseen data. 
   - **Training** was done on groundtruth CSVs:
     - An 80/20 train-validation split.
     - 100 epochs, Adam optimizer, CrossEntropyLoss for multi-class classification.
   - **Inference** on unseen CSVs (test site) grouped each pixel’s 12 monthly records to produce a single predicted class per pixel.

---

## Repository Structure

```
├── README.md                 # This file
└── main-workflow-latest.ipynb          # Main working script
├── Dataset/
│   ├── groundtruth_files/          # groundtruth CSVs
│   ├── GEE_unseen_cleaned_time_series/               # Unlabeled test site CSVs
├── Logs/
│   ├── Time_series_modelling_+_feature_extraction_+_transformer.ipynb      #Transformer attempts, ignore
│   ├── Groundtruth_analysis.ipynb     # Previously ran, ignore 
│   └── ...
└── requirements.txt          # Python dependencies
```

## Technical Highlights

- **Language/Framework:** Python 3.9, PyTorch for the LSTM model, Pandas for data handling.
- **Neural Network Architecture:** LSTM with hidden dimension (e.g., 128), 1–2 layers, dropout 0.2, trained via Adam optimizer.
- **Time-Series Input Shape:** `(batch_size, 12, 20+)` → 12 monthly steps, ~20 spectral features.

---
