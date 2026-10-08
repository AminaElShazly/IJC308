# Anomaly Detection for Greenhouse Digital Twin

**IJC308 Data Science Portfolio - Component 3**

## Overview

This project implements an anomaly detection system for greenhouse sensor data using an ensemble of three machine learning algorithms: Isolation Forest, One-Class SVM, and Local Outlier Factor. A data point is flagged as anomalous only if 2 or more algorithms agree (majority voting).

## Repository Structure

```
├── Anomaly_Detection_Final.ipynb    # Main Jupyter notebook
├── software_component_2_abstract.pdf # Project abstract
├── source_data/                      # Raw data from API
├── results_data/                     # Output with anomaly labels
└── README.md
```

## Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn requests
```

## Reproducing Results

### Option 1: Using Provided Data (Exact Reproduction)

1. Open `Anomaly_Detection_Final.ipynb` in Jupyter or Google Colab

2. Replace the API fetch cell (Section 2) with:
   ```python
   fetched_data = pd.read_csv('source_data/fetched_data.csv')
   ```

3. Run all cells sequentially

4. Compare output with `results_data/anomaly_results.csv` to verify

### Option 2: Using Live API Data

Run the notebook as-is, it will fetch the same data from the Sheffield Greenhouse API (fixed date range: 14 Dec 2025 – 18 Jan 2026).

## Expected Output

- 227 ensemble anomalies detected (2.3% of 10,081 records)
- Individual models each detect ~504 anomalies
- Visualisations: PCA plots, time-series with anomaly markers, correlation heatmaps

## Data Source

- **API**: `http://greenhouse.shef.ac.uk:7070/data`
- **Period**: 14 December 2025 – 18 January 2026 (35 days)
- **Records**: 10,081 sensor readings
- **Features used**: indoor_temperature, indoor_co2, indoor_vpd, outdoor_air_temp, outdoor_rh, outdoor_global_radiation, outdoor_wind_speed

## Notebook Structure

1. **Setup** – Import libraries
2. **Data Collection** – Fetch from Greenhouse API
3. **Data Exploration** – Statistical summaries and distributions
4. **Preprocessing** – Feature selection and normalisation (StandardScaler)
5. **Model Training** – Isolation Forest, One-Class SVM, Local Outlier Factor
6. **Ensemble Voting** – Majority voting (2+ models agree)
7. **Visualisation** – PCA plots, time-series, heatmaps
8. **Analysis** – Compare normal vs anomalous readings
